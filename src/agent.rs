//! The loop. It never calls a model and never runs a tool: it tells you what to do next
//! ([`Step`]) and you tell it what happened. That is the whole contract.

use serde_json::{json, Value};

use crate::recipe::Specialist;
use crate::skill::Skill;
use crate::tool::{Kind, Policy, Risk, Tool, ToolCall, ToolOutput};

/// The name of the tool that hands work to a specialist. Reserved.
pub const DELEGATE: &str = "delegate";
/// The name of the tool that keeps a note for later tasks. Reserved. The agent handles it
/// itself: nothing for your app to run, only [`Agent::new_notes`] to store.
pub const REMEMBER: &str = "remember";

/// The fewest turns (a call and its result are two) added between one summary and the next.
const MIN_TURNS_BETWEEN_SUMMARIES: usize = 6;

/// The name of the tool that loads a skill. Reserved. The agent handles it itself.
pub const USE_SKILL: &str = "use_skill";

/// The most skills one task may load, and the most of a skill's text given back at once.
const MAX_SKILLS_LOADED: usize = 6;
const MAX_SKILL_GIVEN: usize = 12_000;

/// The most notes shown to the model, and the longest note.
const MAX_NOTES_SHOWN: usize = 20;
const MAX_NOTE_CHARS: usize = 300;

/// The knobs. `Config::default()` is a sensible cautious setup.
#[derive(Clone, Debug)]
pub struct Config {
    /// Which kinds of tool need a yes from the person.
    pub policy: Policy,
    /// The most tool calls in one task.
    pub max_calls: u32,
    /// The most steps in a plan.
    pub max_plan_steps: usize,
    /// How long an answer the model may give, passed on in every [`ModelRequest`].
    pub max_tokens: u32,
    /// How much of the history goes back to the model each turn. Older turns are dropped whole.
    pub history_chars: usize,
    /// Ask for a plan first and wait for approval before any tool runs.
    pub require_plan: bool,
    /// How many unusable replies in a row end the task.
    pub bad_reply_limit: u32,
    /// If the model says it is done while most of the plan is untouched, send it back once.
    pub check_plan_before_final: bool,
    /// The most tool calls a specialist may make for one piece of delegated work.
    pub delegate_calls: u32,
    /// When the history outgrows `history_chars`, ask the model to summarise the older steps
    /// (a [`Step::Summarize`]) instead of dropping them. Off by default.
    pub summarize: bool,
    /// Your own words for the model: what the app is, how to behave. Added to the instructions.
    pub instructions: String,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            policy: Policy::default(),
            max_calls: 12,
            max_plan_steps: 8,
            max_tokens: 800,
            history_chars: 9000,
            require_plan: true,
            bad_reply_limit: 3,
            check_plan_before_final: true,
            delegate_calls: 8,
            summarize: false,
            instructions: String::new(),
        }
    }
}

/// What to send to the model. Send `system` and `prompt` as a system message and a user
/// message (or however your model takes them), then give its text to [`Agent::reply`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelRequest {
    pub system: String,
    pub prompt: String,
    pub max_tokens: u32,
}

/// What to do next.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// Send this to the model, then call [`Agent::reply`] with what it said (or [`Agent::fail`]
    /// if the call itself failed).
    Ask(ModelRequest),
    /// The history is long. Send this to the model like any other request, then call
    /// [`Agent::summary`] with its text (or [`Agent::summary_failed`] to drop the old steps
    /// instead). Only happens when [`Config::summarize`] is on.
    Summarize(ModelRequest),
    /// The model proposes this plan. Show it to the person, then call [`Agent::approve_plan`].
    Plan(Vec<String>),
    /// Run this tool, then call [`Agent::result`] with its output. If `call.ask` is true, the
    /// person must say yes first; if they say no, give back [`ToolOutput::declined`].
    Run(ToolCall),
    /// The task is over.
    Done(Outcome),
}

/// Why a task ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The model finished.
    Done,
    /// It used all its calls.
    CallLimit,
    /// The person did not approve the plan.
    Declined,
    /// You called [`Agent::stop`].
    Stopped,
    /// Something went wrong: the model kept giving unusable replies, or a call failed.
    Failed(String),
}

impl Stop {
    pub fn label(&self) -> String {
        match self {
            Stop::Done => "done".into(),
            Stop::CallLimit => "stopped at the call limit".into(),
            Stop::Declined => "plan not approved".into(),
            Stop::Stopped => "stopped".into(),
            Stop::Failed(why) => format!("stopped: {why}"),
        }
    }
}

/// The end of a task.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub stop: Stop,
    /// What the model said it did (empty unless it finished).
    pub summary: String,
    /// How many tool calls were made.
    pub calls: u32,
}

/// One line of the trace, for showing progress or keeping a record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    /// `plan`, `call`, `result`, `retry`, `check`, `final` or `stop`.
    pub kind: &'static str,
    pub text: String,
}

#[derive(Clone, Debug)]
enum Turn {
    Model(String),
    Result(String),
}

#[derive(Clone, Debug)]
enum Phase {
    Idle,
    Model,
    /// Waiting for a summary of this many of the oldest turns.
    Summarize(usize),
    PlanApproval,
    Tool(ToolCall),
    Done(Outcome),
}

/// One task in progress. Make one per task.
pub struct Agent {
    cfg: Config,
    tools: Vec<Tool>,
    specialists: Vec<Specialist>,
    /// The tools the lead model may see and call, when limited (specialists still get theirs).
    lead: Option<Vec<String>>,
    /// Notes kept from earlier tasks, shown to the model, and the ones it added in this one.
    memory: Vec<String>,
    new_notes: Vec<String>,
    memory_on: bool,
    /// Skills the model may load, and the ones it has.
    skills: Vec<Skill>,
    loaded: Vec<String>,
    /// How many turns the history had right after the last summary, so summaries are not
    /// asked for on every step when single results are big.
    since_summary: usize,
    task: String,
    context: String,
    plan: Vec<String>,
    approved: bool,
    confirmed: bool,
    calls: u32,
    next_id: u32,
    bad: u32,
    turns: Vec<Turn>,
    /// What was folded out of the history by earlier summaries.
    summary: String,
    phase: Phase,
    trace: Vec<Event>,
}

enum Action {
    Plan(Vec<String>),
    Tool { name: String, args: Value },
    Final(String),
}

impl Agent {
    pub fn new(cfg: Config, tools: Vec<Tool>) -> Agent {
        let approved = !cfg.require_plan;
        Agent {
            cfg,
            tools,
            specialists: Vec::new(),
            lead: None,
            memory: Vec::new(),
            new_notes: Vec::new(),
            memory_on: false,
            skills: Vec::new(),
            loaded: Vec::new(),
            since_summary: 0,
            task: String::new(),
            context: String::new(),
            plan: Vec::new(),
            approved,
            confirmed: false,
            calls: 0,
            next_id: 1,
            bad: 0,
            turns: Vec::new(),
            summary: String::new(),
            phase: Phase::Idle,
            trace: Vec::new(),
        }
    }

    /// Gives the agent specialists to hand work to. The model gets a `delegate` tool that
    /// names one of them; [`crate::run`] does the rest, or call [`Agent::start_child`] yourself.
    pub fn with_specialists(mut self, specialists: Vec<Specialist>) -> Agent {
        self.tools.retain(|t| t.name != DELEGATE);
        if !specialists.is_empty() {
            let names: Vec<String> = specialists.iter().map(|s| s.name.clone()).collect();
            let list: Vec<String> = specialists.iter().map(|s| format!("{}: {}", s.name, s.help)).collect();
            let known = names.clone();
            self.tools.push(
                Tool::new(DELEGATE, &format!("Hand a piece of the work to a specialist, who does it and reports back. Specialists: {}.", list.join("; ")))
                    .arg("to", Kind::Text, "The specialist's name")
                    .arg("task", Kind::Text, "What to do, in full: the specialist sees only this and the context")
                    .opt("context", Kind::Text, "Anything the specialist needs to know")
                    .risk(Risk::Read)
                    .validate(move |a| {
                        let to = a["to"].as_str().unwrap_or("");
                        if known.iter().any(|n| n == to) { Ok(a.clone()) } else { Err(format!("there is no specialist called {to}; the specialists are {}", known.join(", "))) }
                    }),
            );
        }
        self.specialists = specialists;
        self
    }

    /// Gives the model what was remembered from earlier tasks, and a `remember` tool to keep
    /// more. Store [`Agent::new_notes`] when the task ends and pass them back next time.
    pub fn with_memory(mut self, notes: Vec<String>) -> Agent {
        self.memory = notes;
        self.add_remember_tool();
        self
    }

    fn add_remember_tool(&mut self) {
        self.memory_on = true;
        self.tools.retain(|t| t.name != REMEMBER);
        self.tools.push(
            Tool::new(REMEMBER, "Keep a short note for future tasks: a lasting fact about the person, their preferences or their files. Not for progress on this task.")
                .arg("note", Kind::Text, "The note, one or two sentences")
                .risk(Risk::Create)
                .validate(|a| {
                    let n = a["note"].as_str().unwrap_or("").trim();
                    if n.is_empty() { Err("the note is empty".into()) } else { Ok(serde_json::json!({"note": n.chars().take(MAX_NOTE_CHARS).collect::<String>()})) }
                }),
        );
    }

    /// Gives the model skills it can load with a `use_skill` tool. It is shown only each skill's
    /// name and description; the full text is read when the model asks for it. The agent handles
    /// the tool itself, so there is nothing for your app to run.
    pub fn with_skills(mut self, skills: Vec<Skill>) -> Agent {
        self.skills = skills.into_iter().filter(|s| s.check().is_ok()).collect();
        self.tools.retain(|t| t.name != USE_SKILL);
        if !self.skills.is_empty() {
            self.tools.push(
                Tool::new(USE_SKILL, "Read a skill: instructions for a kind of work, listed below. Load one when its description fits the task, before you start.")
                    .arg("name", Kind::Text, "The skill's name")
                    .risk(Risk::Read),
            );
        }
        self
    }

    /// What was remembered from earlier tasks (and anything added since).
    pub fn memory(&self) -> &[String] {
        &self.memory
    }

    /// The notes the model asked to keep during this task, for your app to store.
    pub fn new_notes(&self) -> &[String] {
        &self.new_notes
    }

    /// Limits what the lead model sees and may call to these tools (and `delegate`). Specialists
    /// still draw their tools from everything registered, so the lead can be kept to looking
    /// and delegating while the specialists do the changing.
    pub fn lead_tools(mut self, names: &[&str]) -> Agent {
        self.lead = Some(names.iter().map(|n| (*n).to_owned()).collect());
        self
    }

    /// The tools the lead model works with.
    fn visible(&self) -> impl Iterator<Item = &Tool> {
        self.tools.iter().filter(move |t| t.name == DELEGATE || t.name == REMEMBER || t.name == USE_SKILL || self.lead.as_ref().is_none_or(|l| l.contains(&t.name)))
    }

    /// For a `delegate` call: a new agent for that specialist, already started, with the first
    /// step. It has only the specialist's tools, needs no plan, and has `delegate_calls` calls.
    /// Run it like any agent, then give the parent `agent.result(call.id, ..)`.
    pub fn start_child(&self, call: &ToolCall) -> Result<(Agent, Step), String> {
        if call.name != DELEGATE {
            return Err(format!("{} is not a delegation", call.name));
        }
        let to = call.args["to"].as_str().unwrap_or("");
        let sp = self.specialists.iter().find(|s| s.name == to).ok_or_else(|| format!("there is no specialist called {to}"))?;
        let tools: Vec<Tool> = self.tools.iter().filter(|t| t.name != DELEGATE && sp.tools.contains(&t.name)).cloned().collect();
        let mut cfg = self.cfg.clone();
        cfg.require_plan = false;
        cfg.check_plan_before_final = false;
        cfg.max_calls = self.cfg.delegate_calls;
        cfg.instructions = [self.cfg.instructions.trim(), sp.instructions.trim()].iter().filter(|s| !s.is_empty()).copied().collect::<Vec<_>>().join("\n\n");
        let mut child = Agent::new(cfg, tools);
        let step = child.start(call.args["task"].as_str().unwrap_or(""), call.args["context"].as_str().unwrap_or(""));
        Ok((child, step))
    }

    /// Begins the task. `context` is whatever the model should know first (what is open, a
    /// document's text). Returns the first step, which is always [`Step::Ask`].
    pub fn start(&mut self, task: &str, context: &str) -> Step {
        self.task = task.trim().to_owned();
        self.context = context.trim().to_owned();
        self.ask()
    }

    /// Gives the model's text. Returns the next step.
    pub fn reply(&mut self, text: &str) -> Step {
        if !matches!(self.phase, Phase::Model) {
            return self.resume();
        }
        match self.read(text) {
            Err(why) => {
                let why = if self.bad >= self.cfg.bad_reply_limit {
                    // The last try failed too: say what the model sent, to help whoever is looking.
                    let snippet: String = text.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(160).collect();
                    format!("{why}; its last reply began: {snippet}")
                } else {
                    why
                };
                self.bad_reply(&why)
            }
            Ok(action) => {
                self.bad = 0;
                self.act(action, text)
            }
        }
    }

    /// The person's answer to a [`Step::Plan`].
    pub fn approve_plan(&mut self, yes: bool) -> Step {
        if !matches!(self.phase, Phase::PlanApproval) {
            return self.resume();
        }
        if !yes {
            return self.finish(Stop::Declined, String::new());
        }
        self.approved = true;
        self.ask()
    }

    /// The output of a [`Step::Run`], with the call's `id`.
    pub fn result(&mut self, id: u32, out: ToolOutput) -> Step {
        let Phase::Tool(call) = self.phase.clone() else {
            return self.resume();
        };
        if call.id != id {
            return self.resume();
        }
        self.turns.push(Turn::Model(json!({"tool": call.name, "args": call.args}).to_string()));
        let text = out.for_model();
        self.trace.push(Event { kind: "result", text: text.clone() });
        self.turns.push(Turn::Result(text));
        self.ask()
    }

    /// The summary the model wrote for a [`Step::Summarize`]. The summarised steps are dropped
    /// from the history and the summary is shown in their place.
    pub fn summary(&mut self, text: &str) -> Step {
        let Phase::Summarize(n) = self.phase else {
            return self.resume();
        };
        let text: String = strip_thinking(text).trim().chars().take(2000).collect();
        if text.is_empty() {
            return self.summary_failed();
        }
        self.trace.push(Event { kind: "summary", text: text.clone() });
        self.turns.drain(..n.min(self.turns.len()));
        self.summary = text;
        self.since_summary = self.turns.len();
        self.ask()
    }

    /// The summary could not be made. The old steps are dropped instead, and the task goes on.
    pub fn summary_failed(&mut self) -> Step {
        let Phase::Summarize(n) = self.phase else {
            return self.resume();
        };
        self.trace.push(Event { kind: "summary", text: "could not be made; the older steps were dropped".into() });
        self.turns.drain(..n.min(self.turns.len()));
        self.since_summary = self.turns.len();
        self.ask()
    }

    /// Your app looked at a finished task and does not accept it (a check failed). Sends the
    /// model back to work with `why` as the latest result. Only works right after [`Step::Done`]
    /// with [`Stop::Done`]; anything else returns the current step unchanged.
    pub fn reject_final(&mut self, why: &str) -> Step {
        let ended_well = matches!(&self.phase, Phase::Done(o) if o.stop == Stop::Done);
        if !ended_well {
            return self.resume();
        }
        self.turns.push(Turn::Result(why.trim().to_owned()));
        self.trace.push(Event { kind: "check", text: why.trim().to_owned() });
        self.ask()
    }

    /// The model call itself failed (network, key, quota). Ends the task.
    pub fn fail(&mut self, why: &str) -> Step {
        self.finish(Stop::Failed(why.to_owned()), String::new())
    }

    /// Ends the task now, for example when the person presses Stop.
    pub fn stop(&mut self) -> Step {
        self.finish(Stop::Stopped, String::new())
    }

    /// The step the agent is waiting on. Safe to call at any time, and what you call after
    /// [`Agent::restore`]. Calling a method at the wrong time also just returns this.
    pub fn resume(&mut self) -> Step {
        match self.phase.clone() {
            Phase::Idle => self.finish(Stop::Failed("the task was never started".into()), String::new()),
            Phase::Model => Step::Ask(self.request()),
            Phase::Summarize(n) => Step::Summarize(self.summary_request(n)),
            Phase::PlanApproval => Step::Plan(self.plan.clone()),
            Phase::Tool(call) => Step::Run(call),
            Phase::Done(o) => Step::Done(o),
        }
    }

    /// The plan the model proposed (empty until it has).
    pub fn plan(&self) -> &[String] {
        &self.plan
    }

    /// How many tool calls have been made.
    pub fn calls(&self) -> u32 {
        self.calls
    }

    /// What has happened so far, in order.
    pub fn trace(&self) -> &[Event] {
        &self.trace
    }

    /// The outcome, once the task is over.
    pub fn outcome(&self) -> Option<&Outcome> {
        match &self.phase {
            Phase::Done(o) => Some(o),
            _ => None,
        }
    }

    // ------------------------------------------------------------ the loop

    fn ask(&mut self) -> Step {
        if self.cfg.summarize {
            if let Some(n) = self.fold_count() {
                self.phase = Phase::Summarize(n);
                return Step::Summarize(self.summary_request(n));
            }
        }
        self.phase = Phase::Model;
        Step::Ask(self.request())
    }

    fn len_of(t: &Turn) -> usize {
        match t {
            Turn::Model(s) | Turn::Result(s) => s.len() + 8,
        }
    }

    /// How many of the oldest turns to fold into a summary: when the history is longer than
    /// allowed, everything but the newest turns that fit in half the allowance (at least two,
    /// and the kept part starts on a model turn so a result is never cut from its call).
    fn fold_count(&self) -> Option<usize> {
        let total: usize = self.turns.iter().map(Agent::len_of).sum::<usize>() + self.summary.len();
        // Not again until a few more steps have been taken: older results are still trimmed
        // from the prompt meanwhile, so a summary is worth a call only now and then.
        if total <= self.cfg.history_chars || self.turns.len() < 4 || self.turns.len() < self.since_summary + MIN_TURNS_BETWEEN_SUMMARIES {
            return None;
        }
        let (mut keep, mut size) = (0, 0);
        for t in self.turns.iter().rev() {
            if size + Agent::len_of(t) > self.cfg.history_chars / 2 && keep >= 2 {
                break;
            }
            size += Agent::len_of(t);
            keep += 1;
        }
        let mut from = self.turns.len() - keep;
        while from > 0 && matches!(self.turns[from], Turn::Result(_)) {
            from -= 1;
        }
        (from > 0).then_some(from)
    }

    fn summary_request(&self, n: usize) -> ModelRequest {
        let mut p = format!("Task: {}\n", self.task);
        if !self.summary.is_empty() {
            p.push_str(&format!("\nSummary so far:\n{}\n", self.summary));
        }
        p.push_str("\nSteps to fold into the summary:\n");
        for t in self.turns.iter().take(n) {
            match t {
                Turn::Model(m) => p.push_str(&format!("You: {m}\n")),
                Turn::Result(r) => p.push_str(&format!("Result: {r}\n")),
            }
        }
        p.push_str("\nWrite the new summary now.");
        ModelRequest {
            system: "You keep notes on a task so work can continue later. Write a short factual summary (under 200 words) of what has been done, what was found, and what is still to do. Keep exact names, numbers, cell references and file names. Reply with the summary only.".into(),
            prompt: p,
            max_tokens: self.cfg.max_tokens,
        }
    }

    fn finish(&mut self, stop: Stop, summary: String) -> Step {
        self.trace.push(Event { kind: "stop", text: stop.label() });
        let o = Outcome { stop, summary, calls: self.calls };
        self.phase = Phase::Done(o.clone());
        Step::Done(o)
    }

    fn bad_reply(&mut self, why: &str) -> Step {
        self.bad += 1;
        self.trace.push(Event { kind: "retry", text: why.to_owned() });
        if self.bad > self.cfg.bad_reply_limit {
            return self.finish(Stop::Failed(format!("the model kept giving replies that could not be used ({why})")), String::new());
        }
        self.turns.push(Turn::Result(format!("Your reply could not be used: {why}. Reply with one JSON object.")));
        self.ask()
    }

    fn act(&mut self, action: Action, raw: &str) -> Step {
        match action {
            Action::Plan(steps) => {
                if !self.cfg.require_plan && self.plan.is_empty() {
                    // Plans are optional here: note it and carry on without asking.
                    self.trace.push(Event { kind: "plan", text: steps.join(" | ") });
                    self.plan = steps;
                    return self.ask();
                }
                if self.approved || !self.plan.is_empty() {
                    self.turns.push(Turn::Model(raw.trim().to_owned()));
                    self.turns.push(Turn::Result("A plan was already approved. Reply with a tool call or the final.".into()));
                    return self.ask();
                }
                if steps.len() > self.cfg.max_plan_steps {
                    return self.bad_reply(&format!("a plan has at most {} steps", self.cfg.max_plan_steps));
                }
                self.trace.push(Event { kind: "plan", text: steps.join(" | ") });
                self.plan = steps.clone();
                self.phase = Phase::PlanApproval;
                Step::Plan(steps)
            }
            Action::Tool { name, args } => {
                if !self.approved {
                    return self.bad_reply("make a plan first");
                }
                if name == USE_SKILL {
                    // Read by the agent itself: it does not count against the call limit.
                    let wanted = args["name"].as_str().unwrap_or("").trim().to_lowercase();
                    let reply = match self.skills.iter().find(|s| s.name == wanted) {
                        None => {
                            let names: Vec<&str> = self.skills.iter().map(|s| s.name.as_str()).collect();
                            format!("There is no skill called {wanted}. The skills are {}.", names.join(", "))
                        }
                        Some(_) if self.loaded.contains(&wanted) => format!("The skill {wanted} is already loaded; its instructions are above."),
                        Some(_) if self.loaded.len() >= MAX_SKILLS_LOADED => format!("Enough skills are loaded ({MAX_SKILLS_LOADED}); carry on with what you have."),
                        Some(s) => {
                            self.loaded.push(wanted.clone());
                            self.trace.push(Event { kind: "skill", text: wanted.clone() });
                            let body: String = s.body.chars().take(MAX_SKILL_GIVEN).collect();
                            format!("Skill {wanted}:\n{body}{}", if s.body.chars().count() > MAX_SKILL_GIVEN { "\n(cut here)" } else { "" })
                        }
                    };
                    self.turns.push(Turn::Model(raw.trim().to_owned()));
                    self.turns.push(Turn::Result(reply));
                    return self.ask();
                }
                if name == REMEMBER {
                    // Kept by the agent itself: no call to run, and it does not count against the limit.
                    let note = args["note"].as_str().unwrap_or("").to_owned();
                    self.trace.push(Event { kind: "note", text: note.clone() });
                    self.memory.push(note.clone());
                    self.new_notes.push(note);
                    self.turns.push(Turn::Model(raw.trim().to_owned()));
                    self.turns.push(Turn::Result("Noted. It will be remembered for future tasks.".into()));
                    return self.ask();
                }
                if self.calls >= self.cfg.max_calls {
                    return self.finish(Stop::CallLimit, String::new());
                }
                let tool = self.visible().find(|t| t.name == name).expect("checked when read");
                let call = ToolCall {
                    id: self.next_id,
                    name,
                    args,
                    risk: tool.risk,
                    ask: self.cfg.policy.asks(tool.risk),
                };
                self.next_id += 1;
                self.calls += 1;
                self.trace.push(Event { kind: "call", text: format!("{} {}", call.name, call.args) });
                self.phase = Phase::Tool(call.clone());
                Step::Run(call)
            }
            Action::Final(summary) => {
                if self.cfg.check_plan_before_final && !self.confirmed && self.approved && (self.calls as usize) + 1 < self.plan.len() {
                    self.confirmed = true;
                    self.trace.push(Event { kind: "check", text: format!("{} calls for a plan of {} steps", self.calls, self.plan.len()) });
                    self.turns.push(Turn::Model(raw.trim().to_owned()));
                    self.turns.push(Turn::Result(format!(
                        "You said you were done, but only {} tool call{} were made for a plan of {} steps. Compare with the approved plan: do any step not yet done, or reply with the final again if everything is done.",
                        self.calls,
                        if self.calls == 1 { "" } else { "s" },
                        self.plan.len()
                    )));
                    return self.ask();
                }
                self.trace.push(Event { kind: "final", text: summary.clone() });
                self.finish(Stop::Done, summary)
            }
        }
    }

    // ------------------------------------------------------------ reading the model

    fn read(&self, text: &str) -> Result<Action, String> {
        let clean = strip_thinking(text);
        let v = first_action_object(&clean)?;
        if let Some(plan) = v.get("plan") {
            let steps: Vec<String> = plan
                .as_array()
                .ok_or("the plan must be a list of steps")?
                .iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_owned())
                .filter(|s| !s.is_empty())
                .collect();
            if steps.is_empty() {
                return Err("a plan needs at least one step".into());
            }
            return Ok(Action::Plan(steps));
        }
        if let Some(f) = v.get("final") {
            return Ok(Action::Final(f.as_str().map(str::to_owned).unwrap_or_else(|| f.to_string())));
        }
        let name = v.get("tool").and_then(Value::as_str).ok_or(r#"the reply named no tool, plan or final. Reply with exactly one of {"plan":["step", ...]}, {"tool":"<name>","args":{...}} or {"final":"what was done"}"#)?;
        let tool = self.visible().find(|t| t.name == name).ok_or_else(|| {
            let names: Vec<&str> = self.visible().map(|t| t.name.as_str()).collect();
            format!("there is no tool called {name}; the tools are {}", names.join(", "))
        })?;
        // Some models put the arguments next to the tool name instead of inside `args`, or split
        // them between the two: take both, with `args` winning.
        let mut given = serde_json::Map::new();
        if let Value::Object(o) = &v {
            for (k, val) in o {
                if k != "tool" && k != "args" {
                    given.insert(k.clone(), val.clone());
                }
            }
        }
        if let Some(Value::Object(a)) = v.get("args") {
            for (k, val) in a {
                given.insert(k.clone(), val.clone());
            }
        }
        let args = tool.check(&Value::Object(given))?;
        Ok(Action::Tool { name: name.to_owned(), args })
    }

    // ------------------------------------------------------------ what the model is told

    fn request(&self) -> ModelRequest {
        ModelRequest { system: self.system(), prompt: self.prompt(), max_tokens: self.cfg.max_tokens }
    }

    fn system(&self) -> String {
        let mut s = String::new();
        if !self.cfg.instructions.trim().is_empty() {
            s.push_str(self.cfg.instructions.trim());
            s.push_str("\n\n");
        }
        s.push_str("You carry out a task one step at a time. Reply with exactly one JSON object and nothing else: no code fences, no explanation.\n");
        if self.cfg.require_plan {
            s.push_str(&format!("First reply: {{\"plan\":[\"short step\", ...]}} with at most {} steps.\nAfter the plan is approved, each reply is one tool call or the end:\n", self.cfg.max_plan_steps));
        } else {
            s.push_str("Each reply is a plan, one tool call, or the end:\n  {\"plan\":[\"short step\", ...]}\n");
        }
        for t in self.visible() {
            s.push_str(&t.describe());
            s.push('\n');
        }
        s.push_str("  {\"final\":\"one or two sentences on what was done\"}\n");
        if !self.skills.is_empty() {
            s.push_str("Skills you can load with use_skill:\n");
            for k in &self.skills {
                s.push_str(&format!("  - {}: {}\n", k.name, k.description.chars().take(400).collect::<String>()));
            }
        }
        s.push_str("Look before you change anything. A change may be held for the person to approve; the next message tells you what happened. If a call fails, fix it or finish and say so.");
        s
    }

    fn prompt(&self) -> String {
        let mut s = format!("Task: {}\n", self.task);
        if !self.context.is_empty() {
            s.push('\n');
            s.push_str(&self.context);
            s.push('\n');
        }
        if !self.memory.is_empty() {
            s.push_str("\nRemembered from before (use it where it matters):\n");
            let from = self.memory.len().saturating_sub(MAX_NOTES_SHOWN);
            for n in &self.memory[from..] {
                s.push_str(&format!("- {}\n", n.chars().take(MAX_NOTE_CHARS).collect::<String>()));
            }
        }
        if self.approved && !self.plan.is_empty() {
            s.push_str("\nApproved plan:\n");
            for (i, p) in self.plan.iter().enumerate() {
                s.push_str(&format!("{}. {p}\n", i + 1));
            }
        }
        if !self.summary.is_empty() {
            s.push_str(&format!("\nSummary of the earlier steps:\n{}\n", self.summary));
        }
        let lines: Vec<String> = self
            .turns
            .iter()
            .map(|t| match t {
                Turn::Model(m) => format!("You: {m}"),
                Turn::Result(r) => format!("Result: {r}"),
            })
            .collect();
        // Keep the newest turns that fit, but never fewer than two, and never start on a result.
        let (mut keep, mut size) = (0, 0);
        for l in lines.iter().rev() {
            if size + l.len() > self.cfg.history_chars && keep >= 2 {
                break;
            }
            size += l.len();
            keep += 1;
        }
        let mut from = lines.len() - keep;
        while from > 0 && matches!(self.turns[from], Turn::Result(_)) {
            from -= 1;
        }
        if from > 0 {
            s.push_str("\n(earlier steps left out)\n");
        }
        if from < lines.len() {
            s.push('\n');
            s.push_str(&lines[from..].join("\n"));
            s.push('\n');
        }
        s.push_str(if self.cfg.require_plan && !self.approved {
            "\nReply with the plan."
        } else {
            "\nReply with the next tool call, or the final."
        });
        s
    }

    // ------------------------------------------------------------ snapshot

    /// The whole state as JSON text. Keep it anywhere (a file, a database row); after a restart
    /// give it to [`Agent::restore`] and call [`Agent::resume`].
    pub fn snapshot(&self) -> String {
        let turns: Vec<Value> = self
            .turns
            .iter()
            .map(|t| match t {
                Turn::Model(m) => json!(["m", m]),
                Turn::Result(r) => json!(["r", r]),
            })
            .collect();
        let (phase, call, outcome) = match &self.phase {
            Phase::Idle => ("idle", Value::Null, Value::Null),
            Phase::Model => ("model", Value::Null, Value::Null),
            Phase::Summarize(n) => ("summarize", json!(n), Value::Null),
            Phase::PlanApproval => ("plan", Value::Null, Value::Null),
            Phase::Tool(c) => ("tool", json!({"id": c.id, "name": c.name, "args": c.args}), Value::Null),
            Phase::Done(o) => ("done", Value::Null, json!({"stop": o.stop.label(), "failed": matches!(o.stop, Stop::Failed(_)), "summary": o.summary})),
        };
        json!({
            "v": 1, "task": self.task, "context": self.context, "plan": self.plan,
            "approved": self.approved, "confirmed": self.confirmed, "calls": self.calls,
            "next_id": self.next_id, "bad": self.bad, "turns": turns, "summary": self.summary, "memory": self.memory, "new_notes": self.new_notes, "memory_on": self.memory_on, "since": self.since_summary, "loaded": self.loaded,
            "phase": phase, "call": call, "outcome": outcome,
        })
        .to_string()
    }

    /// Rebuilds an agent from [`Agent::snapshot`] text. Give it the same tools (and a config).
    /// A task that had ended comes back ended.
    pub fn restore(cfg: Config, tools: Vec<Tool>, snapshot: &str) -> Result<Agent, String> {
        let v: Value = serde_json::from_str(snapshot).map_err(|_| "the snapshot is not JSON".to_string())?;
        if v["v"] != 1 {
            return Err("the snapshot is from another version".into());
        }
        let text = |k: &str| v[k].as_str().unwrap_or("").to_owned();
        let mut a = Agent::new(cfg, tools);
        a.task = text("task");
        a.context = text("context");
        a.plan = v["plan"].as_array().map(|p| p.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect()).unwrap_or_default();
        a.approved = v["approved"].as_bool().unwrap_or(false);
        a.confirmed = v["confirmed"].as_bool().unwrap_or(false);
        a.calls = v["calls"].as_u64().unwrap_or(0) as u32;
        a.next_id = v["next_id"].as_u64().unwrap_or(1) as u32;
        a.bad = v["bad"].as_u64().unwrap_or(0) as u32;
        a.summary = text("summary");
        let notes = |k: &str| -> Vec<String> { v[k].as_array().map(|n| n.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect()).unwrap_or_default() };
        a.memory = notes("memory");
        a.new_notes = notes("new_notes");
        a.since_summary = v["since"].as_u64().unwrap_or(0) as usize;
        a.loaded = v["loaded"].as_array().map(|n| n.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect()).unwrap_or_default();
        if v["memory_on"].as_bool().unwrap_or(false) {
            a.add_remember_tool();
        }
        for t in v["turns"].as_array().into_iter().flatten() {
            let body = t[1].as_str().unwrap_or("").to_owned();
            a.turns.push(if t[0] == "m" { Turn::Model(body) } else { Turn::Result(body) });
        }
        a.phase = match v["phase"].as_str().unwrap_or("") {
            "model" => Phase::Model,
            "summarize" => Phase::Summarize(v["call"].as_u64().unwrap_or(0) as usize),
            "plan" => Phase::PlanApproval,
            "tool" => {
                let name = v["call"]["name"].as_str().unwrap_or("");
                // A delegation is not a registered tool until specialists are given again.
                let risk = if name == DELEGATE {
                    Risk::Read
                } else {
                    a.tools.iter().find(|t| t.name == name).ok_or_else(|| format!("the snapshot needs the tool {name}, which is not registered"))?.risk
                };
                Phase::Tool(ToolCall {
                    id: v["call"]["id"].as_u64().unwrap_or(0) as u32,
                    name: name.to_owned(),
                    args: v["call"]["args"].clone(),
                    risk,
                    ask: a.cfg.policy.asks(risk),
                })
            }
            "done" => {
                let label = v["outcome"]["stop"].as_str().unwrap_or("done").to_owned();
                let stop = if v["outcome"]["failed"].as_bool().unwrap_or(false) {
                    Stop::Failed(label.trim_start_matches("stopped: ").to_owned())
                } else {
                    match label.as_str() {
                        "stopped at the call limit" => Stop::CallLimit,
                        "plan not approved" => Stop::Declined,
                        "stopped" => Stop::Stopped,
                        _ => Stop::Done,
                    }
                };
                Phase::Done(Outcome { stop, summary: v["outcome"]["summary"].as_str().unwrap_or("").to_owned(), calls: a.calls })
            }
            _ => Phase::Idle,
        };
        Ok(a)
    }
}

/// The first JSON object in the text that looks like an answer (it has `plan`, `tool` or
/// `final`), even when the model wrote words around it or a second object after it. Falls back
/// to the first object that parses at all.
fn first_action_object(text: &str) -> Result<Value, String> {
    let bytes = text.as_bytes();
    let mut first_ok: Option<Value> = None;
    let mut cut = false;
    let mut from = 0;
    while let Some(off) = text[from..].find('{') {
        let start = from + off;
        // Find the matching close, skipping braces inside strings.
        let (mut depth, mut in_str, mut esc, mut end) = (0usize, false, false, None);
        for (i, &b) in bytes.iter().enumerate().skip(start) {
            if in_str {
                if esc {
                    esc = false;
                } else if b == b'\\' {
                    esc = true;
                } else if b == b'"' {
                    in_str = false;
                }
                continue;
            }
            match b {
                b'"' => in_str = true,
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }
        match end {
            Some(e) => {
                if let Ok(v) = serde_json::from_str::<Value>(&text[start..=e]) {
                    if v.get("plan").is_some() || v.get("tool").is_some() || v.get("final").is_some() {
                        return Ok(v);
                    }
                    first_ok.get_or_insert(v);
                }
                from = start + 1;
            }
            None => {
                cut = true;
                break;
            }
        }
    }
    first_ok.ok_or_else(|| {
        if cut {
            "the reply was cut off before its JSON object ended, so it was too long. Write less in one call (for example fewer or shorter blocks) and make another call for the rest".to_string()
        } else if text.contains('{') {
            "the reply was not valid JSON".to_string()
        } else {
            "the reply had no JSON object in it".to_string()
        }
    })
}

/// Removes the thinking some models put in front of an answer (`<think>...</think>`).
fn strip_thinking(text: &str) -> String {
    let mut out = text.to_owned();
    while let (Some(a), Some(b)) = (out.find("<think>"), out.find("</think>")) {
        if b < a {
            break;
        }
        out.replace_range(a..b + "</think>".len(), "");
    }
    out
}
