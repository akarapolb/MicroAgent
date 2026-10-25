use microagent::{run, Agent, Config, Host, Kind, ModelRequest, Policy, Recipe, Risk, Specialist, Step, Stop, Tool, ToolCall, ToolOutput};
use serde_json::json;

fn tools() -> Vec<Tool> {
    vec![
        Tool::new("read", "Read a note").arg("name", Kind::Text, "Which note").risk(Risk::Read),
        Tool::new("save", "Save a note")
            .arg("name", Kind::Text, "Which note")
            .arg("text", Kind::Text, "Its text")
            .opt("version", Kind::Number, "Version")
            .risk(Risk::Write),
        Tool::new("email", "Send an email").arg("to", Kind::Text, "Address").risk(Risk::Outside),
    ]
}

/// A host whose model says what the script says, and which records what happened.
struct Script {
    replies: Vec<String>,
    asked: Vec<ModelRequest>,
    ran: Vec<String>,
    approve_plan: bool,
    approve_calls: bool,
    asked_about: Vec<String>,
}

impl Script {
    fn new(replies: &[&str]) -> Script {
        Script {
            replies: replies.iter().rev().map(|s| s.to_string()).collect(),
            asked: vec![],
            ran: vec![],
            approve_plan: true,
            approve_calls: true,
            asked_about: vec![],
        }
    }
}

impl Host for Script {
    fn ask_model(&mut self, req: &ModelRequest) -> Result<String, String> {
        self.asked.push(req.clone());
        self.replies.pop().ok_or_else(|| "the script ran out".to_string())
    }
    fn run_tool(&mut self, call: &ToolCall) -> ToolOutput {
        self.ran.push(format!("{} {}", call.name, call.args));
        if call.name == "read" {
            ToolOutput::ok("the note says hello")
        } else {
            ToolOutput::ok("done")
        }
    }
    fn approve_plan(&mut self, _: &[String]) -> bool {
        self.approve_plan
    }
    fn approve_call(&mut self, call: &ToolCall) -> bool {
        self.asked_about.push(call.name.clone());
        self.approve_calls
    }
}

fn trusting() -> Config {
    Config { policy: Policy::TRUSTING, ..Config::default() }
}

#[test]
fn a_task_is_planned_run_and_finished() {
    let mut host = Script::new(&[
        r#"{"plan":["read the note","save a copy"]}"#,
        r#"{"tool":"read","args":{"name":"a"}}"#,
        r#"{"tool":"save","args":{"name":"b","text":"hello"}}"#,
        r#"{"final":"Copied note a to b."}"#,
    ]);
    let mut agent = Agent::new(trusting(), tools());
    let out = run(&mut agent, &mut host, "Copy note a to b", "Notes: a, b");
    assert_eq!(out.stop, Stop::Done);
    assert_eq!((out.summary.as_str(), out.calls), ("Copied note a to b.", 2));
    assert_eq!(host.ran, vec![r#"read {"name":"a"}"#, r#"save {"name":"b","text":"hello"}"#]);
    // The model is told the tools, the task, the context, and what happened before.
    let first = &host.asked[0];
    assert!(first.system.contains("read") && first.system.contains("save") && first.system.contains("[changes things]"));
    assert!(first.prompt.contains("Task: Copy note a to b") && first.prompt.contains("Notes: a, b") && first.prompt.ends_with("Reply with the plan."));
    let last = &host.asked[3];
    assert!(last.prompt.contains("1. read the note") && last.prompt.contains("Result: the note says hello"));
}

#[test]
fn nothing_runs_before_the_plan_is_approved_and_no_declines_it() {
    let mut host = Script::new(&[r#"{"plan":["do it"]}"#]);
    host.approve_plan = false;
    let out = run(&mut Agent::new(trusting(), tools()), &mut host, "t", "");
    assert_eq!(out.stop, Stop::Declined);
    assert!(host.ran.is_empty());
}

#[test]
fn a_tool_before_a_plan_is_sent_back() {
    let mut host = Script::new(&[
        r#"{"tool":"read","args":{"name":"a"}}"#,
        r#"{"plan":["read"]}"#,
        r#"{"tool":"read","args":{"name":"a"}}"#,
        r#"{"final":"ok"}"#,
    ]);
    let out = run(&mut Agent::new(trusting(), tools()), &mut host, "t", "");
    assert_eq!(out.stop, Stop::Done);
    assert!(host.asked[1].prompt.contains("make a plan first"));
}

#[test]
fn the_policy_decides_which_calls_need_a_yes() {
    let script = [
        r#"{"plan":["a","b","c"]}"#,
        r#"{"tool":"read","args":{"name":"a"}}"#,
        r#"{"tool":"save","args":{"name":"a","text":"x"}}"#,
        r#"{"tool":"email","args":{"to":"me@example.com"}}"#,
        r#"{"final":"ok"}"#,
    ];
    // Cautious: the write and the email ask; the read does not.
    let mut host = Script::new(&script);
    run(&mut Agent::new(Config::default(), tools()), &mut host, "t", "");
    assert_eq!(host.asked_about, vec!["save", "email"]);
    // Trusting: only the email (outside the app) asks.
    let mut host = Script::new(&script);
    run(&mut Agent::new(trusting(), tools()), &mut host, "t", "");
    assert_eq!(host.asked_about, vec!["email"]);
}

#[test]
fn a_refused_call_does_not_run_and_the_model_is_told() {
    let mut host = Script::new(&[
        r#"{"plan":["write"]}"#,
        r#"{"tool":"save","args":{"name":"a","text":"x"}}"#,
        r#"{"final":"I could not save it."}"#,
    ]);
    host.approve_calls = false;
    let out = run(&mut Agent::new(Config::default(), tools()), &mut host, "t", "");
    assert!(host.ran.is_empty());
    assert!(host.asked[2].prompt.contains("The person declined"));
    assert_eq!(out.stop, Stop::Done);
}

#[test]
fn money_always_asks_even_when_the_policy_says_no() {
    let cfg = Config {
        policy: Policy { read: false, create: false, write: false, outside: false, money: false },
        require_plan: false,
        ..Config::default()
    };
    let mut agent = Agent::new(cfg, vec![Tool::new("pay", "Pay").arg("amount", Kind::Number, "How much").risk(Risk::Money)]);
    agent.start("t", "");
    let Step::Run(call) = agent.reply(r#"{"tool":"pay","args":{"amount":"12.5"}}"#) else { panic!() };
    assert!(call.ask);
    assert_eq!(call.args, json!({"amount": 12.5}));
}

#[test]
fn arguments_are_checked_and_cleaned_before_the_host_sees_them() {
    let cfg = || Config { require_plan: false, ..Config::default() };
    let mut agent = Agent::new(cfg(), tools());
    agent.start("t", "");
    // A number given for a text, a text for a number, and an unknown extra.
    let Step::Run(call) = agent.reply(r#"{"tool":"save","args":{"name":7,"text":"x","version":"2","junk":1}}"#) else { panic!() };
    assert_eq!(call.args, json!({"name": "7", "text": "x", "version": 2}));
    // A missing required one is a bad reply, which the model is told about.
    let mut agent = Agent::new(cfg(), tools());
    agent.start("t", "");
    let Step::Ask(req) = agent.reply(r#"{"tool":"save","args":{"name":"a"}}"#) else { panic!() };
    assert!(req.prompt.contains("args.text is missing"), "{}", req.prompt);
}

#[test]
fn unusable_replies_are_retried_and_then_end_the_task() {
    let mut host = Script::new(&["I would love to help!", "Sure.", "Here you go.", "never reached"]);
    let out = run(&mut Agent::new(trusting(), tools()), &mut host, "t", "");
    assert!(matches!(out.stop, Stop::Failed(ref w) if w.contains("kept giving")), "{:?}", out.stop);
    assert_eq!(host.asked.len(), 4);
    // A good reply in between starts the count again.
    let mut host = Script::new(&["nope", r#"{"plan":["x"]}"#, "nope again", r#"{"final":"ok"}"#]);
    assert_eq!(run(&mut Agent::new(trusting(), tools()), &mut host, "t", "").stop, Stop::Done);
}

#[test]
fn fences_and_thinking_around_the_json_are_ignored() {
    let mut host = Script::new(&["<think>hmm {not json}</think>\n```json\n{\"plan\":[\"x\"]}\n```", "Okay: {\"final\":\"ok\"} thanks"]);
    assert_eq!(run(&mut Agent::new(trusting(), tools()), &mut host, "t", "").stop, Stop::Done);
}

#[test]
fn the_call_limit_ends_a_task() {
    let mut script = vec![r#"{"plan":["loop"]}"#.to_string()];
    script.extend(std::iter::repeat_n(r#"{"tool":"read","args":{"name":"a"}}"#.to_string(), 10));
    let refs: Vec<&str> = script.iter().map(String::as_str).collect();
    let mut host = Script::new(&refs);
    let cfg = Config { max_calls: 3, ..trusting() };
    let out = run(&mut Agent::new(cfg, tools()), &mut host, "t", "");
    assert_eq!((out.stop, out.calls, host.ran.len()), (Stop::CallLimit, 3, 3));
}

#[test]
fn done_too_early_is_sent_back_once() {
    let mut host = Script::new(&[
        r#"{"plan":["a","b","c","d"]}"#,
        r#"{"tool":"read","args":{"name":"a"}}"#,
        r#"{"final":"All done."}"#,
        r#"{"final":"Really done."}"#,
    ]);
    let out = run(&mut Agent::new(trusting(), tools()), &mut host, "t", "");
    assert_eq!(out.summary, "Really done.");
    assert!(host.asked[3].prompt.contains("plan of 4 steps"));
}

#[test]
fn a_failed_model_call_ends_the_task_with_the_reason() {
    let mut host = Script::new(&[]); // the first ask fails: the script has no reply
    let out = run(&mut Agent::new(trusting(), tools()), &mut host, "t", "");
    assert!(matches!(out.stop, Stop::Failed(ref w) if w.contains("ran out")));
}

#[test]
fn old_turns_are_dropped_whole_when_the_history_is_long() {
    let cfg = Config { require_plan: false, history_chars: 2000, max_calls: 30, ..Config::default() };
    let mut agent = Agent::new(cfg, tools());
    let mut step = agent.start("t", "");
    for i in 0..30 {
        let Step::Ask(_) = step else { panic!() };
        let Step::Run(call) = agent.reply(&format!(r#"{{"tool":"read","args":{{"name":"n{i}"}}}}"#)) else { panic!() };
        step = agent.result(call.id, ToolOutput::ok("y".repeat(500)));
    }
    let Step::Ask(req) = step else { panic!() };
    assert!(req.prompt.len() < 4000, "{}", req.prompt.len());
    assert!(req.prompt.contains("earlier steps left out") && req.prompt.contains("n29"));
    let first = req.prompt.lines().find(|l| l.starts_with("You:") || l.starts_with("Result:")).unwrap();
    assert!(first.starts_with("You:"), "{first}");
}

#[test]
fn a_task_can_be_snapshotted_and_resumed_after_a_restart() {
    let mut agent = Agent::new(trusting(), tools());
    agent.start("Copy a note", "ctx");
    let Step::Plan(plan) = agent.reply(r#"{"plan":["read","save"]}"#) else { panic!() };
    assert_eq!(plan.len(), 2);
    // Restart while waiting for the plan to be approved.
    let mut agent = Agent::restore(trusting(), tools(), &agent.snapshot()).unwrap();
    assert!(matches!(agent.resume(), Step::Plan(_)));
    assert!(matches!(agent.approve_plan(true), Step::Ask(_)));
    let Step::Run(call) = agent.reply(r#"{"tool":"read","args":{"name":"a"}}"#) else { panic!() };
    // Restart while a tool is waiting to be run: the same call comes back with the same id.
    let mut agent = Agent::restore(trusting(), tools(), &agent.snapshot()).unwrap();
    let Step::Run(again) = agent.resume() else { panic!() };
    assert_eq!(again, call);
    let Step::Ask(req) = agent.result(again.id, ToolOutput::ok("hi")) else { panic!() };
    assert!(req.prompt.contains("Approved plan") && req.prompt.contains("Result: hi") && req.prompt.contains("Task: Copy a note"));
    let Step::Done(o) = agent.reply(r#"{"final":"ok"}"#) else { panic!() };
    // A finished task restores as finished.
    let back = Agent::restore(trusting(), tools(), &agent.snapshot()).unwrap();
    assert_eq!(back.outcome(), Some(&o));
    // A snapshot that waits on a tool you did not register says so.
    let mut waiting = Agent::new(trusting(), tools());
    waiting.start("t", "");
    waiting.reply(r#"{"plan":["x"]}"#);
    waiting.approve_plan(true);
    waiting.reply(r#"{"tool":"read","args":{"name":"a"}}"#);
    let err = Agent::restore(trusting(), vec![], &waiting.snapshot()).err().unwrap();
    assert!(err.contains("read"), "{err}");
}

#[test]
fn a_call_at_the_wrong_time_just_returns_what_the_agent_is_waiting_for() {
    // Not started: there is nothing to reply to.
    let mut agent = Agent::new(trusting(), tools());
    assert!(matches!(agent.reply("{}"), Step::Done(_)));
    let mut agent = Agent::new(trusting(), tools());
    let asking = agent.start("t", "");
    assert_eq!(agent.resume(), asking);
    assert_eq!(agent.approve_plan(true), asking, "no plan is waiting");
    assert_eq!(agent.result(99, ToolOutput::ok("x")), asking, "no call is waiting");
    let Step::Plan(_) = agent.reply(r#"{"plan":["x"]}"#) else { panic!() };
    assert!(matches!(agent.reply(r#"{"final":"x"}"#), Step::Plan(_)), "a reply while a plan is waiting changes nothing");
}

#[test]
fn stop_ends_a_task_and_the_trace_says_what_happened() {
    let mut agent = Agent::new(trusting(), tools());
    agent.start("t", "");
    agent.reply(r#"{"plan":["a"]}"#);
    agent.approve_plan(true);
    let Step::Run(call) = agent.reply(r#"{"tool":"read","args":{"name":"a"}}"#) else { panic!() };
    agent.result(call.id, ToolOutput::error("no such note"));
    let Step::Done(o) = agent.stop() else { panic!() };
    assert_eq!(o.stop, Stop::Stopped);
    let kinds: Vec<&str> = agent.trace().iter().map(|e| e.kind).collect();
    assert_eq!(kinds, vec!["plan", "call", "result", "stop"]);
    assert!(agent.trace()[2].text.starts_with("Failed: no such note"));
}

#[test]
fn garbage_never_panics() {
    let long = "{".repeat(10_000);
    for junk in ["", "{", "}{", "{}", "{\"tool\":5}", "{\"plan\":5}", "{\"plan\":[]}", "{\"tool\":\"read\"}", "\u{0}", "{\"final\":null}", "[1,2]", long.as_str()] {
        let mut agent = Agent::new(trusting(), tools());
        agent.start("t", "");
        let _ = agent.reply(junk);
        let _ = agent.resume();
    }
    assert!(Agent::restore(trusting(), tools(), "nonsense").is_err());
    assert!(Agent::restore(trusting(), tools(), "{\"v\":2}").is_err());
}

#[test]
fn new_things_run_at_once_even_when_changes_ask() {
    let mut agent = Agent::new(
        Config { require_plan: false, ..Config::default() },
        vec![Tool::new("new_note", "Make a note").arg("name", Kind::Text, "Name").risk(Risk::Create)],
    );
    agent.start("t", "");
    let Step::Run(call) = agent.reply(r#"{"tool":"new_note","args":{"name":"a"}}"#) else { panic!() };
    assert!(!call.ask);
    assert!(!Policy::CAUTIOUS.asks(Risk::Create) && Policy::CAUTIOUS.asks(Risk::Write));
}

#[test]
fn a_tool_can_add_its_own_check_of_the_arguments() {
    let t = Tool::new("set", "Set cells").arg("cells", Kind::List, "Cells").risk(Risk::Write).validate(|a| {
        let n = a["cells"].as_array().map_or(0, Vec::len);
        if n > 2 { Err(format!("write at most 2 cells, not {n}")) } else { Ok(a.clone()) }
    });
    let mut agent = Agent::new(Config { require_plan: false, ..Config::default() }, vec![t]);
    agent.start("t", "");
    let Step::Ask(req) = agent.reply(r#"{"tool":"set","args":{"cells":[1,2,3]}}"#) else { panic!() };
    assert!(req.prompt.contains("write at most 2 cells, not 3"));
    assert!(matches!(agent.reply(r#"{"tool":"set","args":{"cells":[1]}}"#), Step::Run(_)));
}

#[test]
fn a_finish_your_app_does_not_accept_sends_the_model_back_to_work() {
    let mut agent = Agent::new(Config { require_plan: false, ..Config::default() }, tools());
    agent.start("t", "");
    let Step::Done(o) = agent.reply(r#"{"final":"done"}"#) else { panic!() };
    assert_eq!(o.stop, Stop::Done);
    let Step::Ask(req) = agent.reject_final("Cell B2 shows an error. Fix it.") else { panic!() };
    assert!(req.prompt.contains("Cell B2 shows an error"));
    let Step::Done(o) = agent.reply(r#"{"final":"fixed"}"#) else { panic!() };
    assert_eq!(o.summary, "fixed");
    // Rejecting a task that did not finish well changes nothing.
    let mut stopped = Agent::new(Config::default(), tools());
    stopped.start("t", "");
    stopped.stop();
    assert!(matches!(stopped.reject_final("no"), Step::Done(_)));
}

fn long_run(summarize: bool) -> (Agent, Step) {
    let cfg = Config { require_plan: false, summarize, history_chars: 1500, max_calls: 40, ..Config::default() };
    let mut agent = Agent::new(cfg, tools());
    let mut step = agent.start("Audit the notes", "");
    for i in 0..12 {
        let Step::Ask(_) = step else { return (agent, step) };
        let Step::Run(call) = agent.reply(&format!(r#"{{"tool":"read","args":{{"name":"note{i}"}}}}"#)) else { panic!() };
        step = agent.result(call.id, ToolOutput::ok(format!("note{i} says {}", "x".repeat(300))));
    }
    (agent, step)
}

#[test]
fn a_long_history_is_summarised_instead_of_dropped_when_asked_to() {
    let (mut agent, step) = long_run(true);
    let Step::Summarize(req) = step else { panic!("{step:?}") };
    let last = format!("note{}", agent.calls() - 1);
    assert!(req.prompt.contains("Task: Audit the notes") && req.prompt.contains("Steps to fold into the summary") && req.prompt.contains("note0"));
    assert!(req.system.contains("short factual summary"));
    let Step::Ask(next) = agent.summary("Read notes 0 to 8; each said x. Nothing wrong found yet.") else { panic!() };
    assert!(next.prompt.contains("Summary of the earlier steps") && next.prompt.contains("Nothing wrong found yet"));
    assert!(!next.prompt.contains("note0 says"), "the folded steps are gone");
    assert!(next.prompt.contains(&last), "the newest steps are kept");
    assert!(agent.trace().iter().any(|e| e.kind == "summary"));
}

#[test]
fn without_the_switch_nothing_is_ever_summarised() {
    let (_, step) = long_run(false);
    assert!(matches!(step, Step::Ask(_)));
}

#[test]
fn a_failed_summary_drops_the_old_steps_and_the_task_goes_on() {
    let (mut agent, step) = long_run(true);
    assert!(matches!(step, Step::Summarize(_)));
    let last = format!("note{}", agent.calls() - 1);
    let Step::Ask(next) = agent.summary_failed() else { panic!() };
    assert!(!next.prompt.contains("note0 says") && next.prompt.contains(&last));
    // An empty summary counts as a failure.
    let (mut agent, _) = long_run(true);
    assert!(matches!(agent.summary("  "), Step::Ask(_)));
}

#[test]
fn a_summary_is_folded_into_the_next_one_and_survives_a_restart() {
    // Waiting for a summary when the app closes: the same request comes back.
    let (agent2, step) = long_run(true);
    let Step::Summarize(before) = step else { panic!() };
    let cfg = Config { require_plan: false, summarize: true, history_chars: 1500, max_calls: 40, ..Config::default() };
    let mut restored = Agent::restore(cfg.clone(), tools(), &agent2.snapshot()).unwrap();
    let Step::Summarize(after) = restored.resume() else { panic!() };
    assert_eq!(before, after);
    restored.summary("Kept summary.");
    let mut again = Agent::restore(cfg, tools(), &restored.snapshot()).unwrap();
    let Step::Ask(req) = again.resume() else { panic!() };
    assert!(req.prompt.contains("Kept summary."));
}

#[test]
fn run_asks_the_host_for_summaries_with_its_ordinary_model_call() {
    struct Notes { asked: Vec<String>, calls: u32 }
    impl Host for Notes {
        fn ask_model(&mut self, req: &ModelRequest) -> Result<String, String> {
            self.asked.push(req.system.clone());
            if req.system.contains("short factual summary") {
                return Ok("Summary: reading notes.".into());
            }
            self.calls += 1;
            if self.calls <= 12 { Ok(format!(r#"{{"tool":"read","args":{{"name":"n{}"}}}}"#, self.calls)) } else { Ok(r#"{"final":"done"}"#.into()) }
        }
        fn run_tool(&mut self, _: &ToolCall) -> ToolOutput {
            ToolOutput::ok("y".repeat(300))
        }
    }
    let cfg = Config { require_plan: false, summarize: true, history_chars: 1500, max_calls: 40, ..Config::default() };
    let mut host = Notes { asked: vec![], calls: 0 };
    let out = run(&mut Agent::new(cfg, tools()), &mut host, "t", "");
    assert_eq!(out.stop, Stop::Done);
    assert!(host.asked.iter().any(|s| s.contains("short factual summary")), "the host was asked to summarise");
}

fn crew() -> Vec<Specialist> {
    vec![
        Specialist::new("reader", "Looks things up").tools(&["read"]).instructions("Only look; never change anything."),
        Specialist::new("writer", "Saves notes").tools(&["save"]),
    ]
}

#[test]
fn a_specialist_does_a_piece_of_the_work_and_reports_back() {
    let mut host = Script::new(&[
        r#"{"plan":["ask the reader","say so"]}"#,
        r#"{"tool":"delegate","args":{"to":"reader","task":"Read note a and tell me what it says","context":"notes: a"}}"#,
        // the reader's own turns
        r#"{"tool":"read","args":{"name":"a"}}"#,
        r#"{"final":"Note a says hello."}"#,
        // back in the parent
        r#"{"final":"The reader found: hello."}"#,
    ]);
    let mut agent = Agent::new(trusting(), tools()).with_specialists(crew());
    let out = run(&mut agent, &mut host, "What does note a say?", "");
    assert_eq!(out.stop, Stop::Done);
    assert_eq!(out.calls, 1, "the parent made one call: the delegation");
    assert_eq!(host.ran, vec![r#"read {"name":"a"}"#]);
    // The parent is told about the specialists; the child is told its own instructions and sees only its tools.
    assert!(host.asked[0].system.contains("delegate") && host.asked[0].system.contains("reader: Looks things up"));
    let child = &host.asked[2];
    assert!(child.system.contains("Only look; never change anything.") && child.system.contains("\"tool\":\"read\""));
    assert!(!child.system.contains("\"tool\":\"save\"") && !child.system.contains("delegate"));
    assert!(child.prompt.contains("Task: Read note a and tell me what it says") && child.prompt.contains("notes: a"));
    // The parent gets the child's summary as the result.
    assert!(host.asked[4].prompt.contains("Result: Note a says hello."));
}

#[test]
fn a_specialist_cannot_use_a_tool_it_was_not_given() {
    let mut host = Script::new(&[
        r#"{"plan":["x"]}"#,
        r#"{"tool":"delegate","args":{"to":"reader","task":"Save a note"}}"#,
        r#"{"tool":"save","args":{"name":"a","text":"x"}}"#,
        r#"{"final":"I could not."}"#,
        r#"{"final":"ok"}"#,
    ]);
    run(&mut Agent::new(trusting(), tools()).with_specialists(crew()), &mut host, "t", "");
    assert!(host.ran.is_empty(), "the reader has no save tool");
    assert!(host.asked[3].prompt.contains("there is no tool called save"));
}

#[test]
fn an_unknown_specialist_is_sent_back_and_children_have_a_call_limit() {
    let mut agent = Agent::new(Config { require_plan: false, ..Config::default() }, tools()).with_specialists(crew());
    agent.start("t", "");
    let Step::Ask(req) = agent.reply(r#"{"tool":"delegate","args":{"to":"nobody","task":"x"}}"#) else { panic!() };
    assert!(req.prompt.contains("there is no specialist called nobody"));
    // A child may make only `delegate_calls` calls.
    let mut agent = Agent::new(Config { require_plan: false, delegate_calls: 2, ..Config::default() }, tools()).with_specialists(crew());
    agent.start("t", "");
    let Step::Run(call) = agent.reply(r#"{"tool":"delegate","args":{"to":"reader","task":"look"}}"#) else { panic!() };
    let (mut child, first) = agent.start_child(&call).unwrap();
    assert!(matches!(first, Step::Ask(_)));
    for _ in 0..2 {
        let Step::Run(c) = child.reply(r#"{"tool":"read","args":{"name":"a"}}"#) else { panic!() };
        child.result(c.id, ToolOutput::ok("x"));
    }
    assert!(matches!(child.reply(r#"{"tool":"read","args":{"name":"a"}}"#), Step::Done(o) if o.stop == Stop::CallLimit));
    // Only a delegation can start a child.
    assert!(agent.start_child(&ToolCall { id: 9, name: "read".into(), args: json!({}), risk: Risk::Read, ask: false }).is_err());
}

#[test]
fn a_specialists_changes_still_ask_under_the_cautious_policy() {
    let mut host = Script::new(&[
        r#"{"plan":["x"]}"#,
        r#"{"tool":"delegate","args":{"to":"writer","task":"Save note b"}}"#,
        r#"{"tool":"save","args":{"name":"b","text":"hi"}}"#,
        r#"{"final":"Saved."}"#,
        r#"{"final":"ok"}"#,
    ]);
    host.approve_calls = false;
    run(&mut Agent::new(Config::default(), tools()).with_specialists(crew()), &mut host, "t", "");
    assert_eq!(host.asked_about, vec!["save"]);
    assert!(host.ran.is_empty());
}

#[test]
fn a_recipe_fills_its_blanks_filters_tools_and_travels_as_json() {
    let r = Recipe::new("monthly", "Summarise spending for {month} and flag anything over {limit}. Again, {month}.")
        .param("month", "Which month")
        .param("limit", "The limit")
        .only_tools(&["read"]);
    assert_eq!(r.blanks(), vec!["month", "limit"]);
    assert_eq!(r.fill(&[("month", "September"), ("limit", "50,000")]).unwrap(), "Summarise spending for September and flag anything over 50,000. Again, September.");
    assert_eq!(r.fill(&[("month", "September")]).unwrap_err(), "limit needs a value");
    assert!(r.fill(&[("month", "  "), ("limit", "1")]).is_err());
    let names: Vec<String> = r.tools_for(&tools()).iter().map(|t| t.name.clone()).collect();
    assert_eq!(names, vec!["read"]);
    assert_eq!(Recipe::new("all", "t").tools_for(&tools()).len(), 3);
    assert_eq!(Recipe::from_json(&r.to_json()).unwrap(), r);
    assert!(Recipe::from_json("nope").is_err() && Recipe::from_json(r#"{"name":"x"}"#).is_err());
    // Braces that are not blanks are left alone.
    assert_eq!(Recipe::new("j", r#"Write {"a": 1} for {who}"#).fill(&[("who", "me")]).unwrap(), r#"Write {"a": 1} for me"#);
}

#[test]
fn the_lead_can_be_kept_to_looking_and_delegating_while_specialists_change_things() {
    let mut agent = Agent::new(Config { require_plan: false, ..Config::default() }, tools()).with_specialists(crew()).lead_tools(&["read"]);
    let Step::Ask(req) = agent.start("t", "") else { panic!() };
    assert!(req.system.contains("\"tool\":\"read\"") && req.system.contains("delegate"));
    assert!(!req.system.contains("\"tool\":\"save\"") && !req.system.contains("\"tool\":\"email\""));
    // The lead cannot call a tool it was not shown...
    let Step::Ask(again) = agent.reply(r#"{"tool":"save","args":{"name":"a","text":"x"}}"#) else { panic!() };
    assert!(again.prompt.contains("there is no tool called save"));
    // ...but the writer specialist can use it.
    let Step::Run(call) = agent.reply(r#"{"tool":"delegate","args":{"to":"writer","task":"save a"}}"#) else { panic!() };
    let (mut child, _) = agent.start_child(&call).unwrap();
    assert!(matches!(child.reply(r#"{"tool":"save","args":{"name":"a","text":"x"}}"#), Step::Run(_)));
}

#[test]
fn the_model_can_keep_notes_for_later_and_is_shown_old_ones() {
    let mut agent = Agent::new(Config { require_plan: false, ..Config::default() }, tools()).with_memory(vec!["The person prefers short answers.".into()]);
    let Step::Ask(req) = agent.start("t", "") else { panic!() };
    assert!(req.prompt.contains("Remembered from before") && req.prompt.contains("The person prefers short answers."));
    assert!(req.system.contains("\"tool\":\"remember\"") && req.system.contains("[makes new things]"));
    // Remembering is handled inside: no call for the host, and it is not counted.
    let Step::Ask(next) = agent.reply(r#"{"tool":"remember","args":{"note":"  Reports go out on Fridays.  "}}"#) else { panic!() };
    assert!(next.prompt.contains("Noted.") && next.prompt.contains("- Reports go out on Fridays."));
    assert_eq!((agent.calls(), agent.new_notes().to_vec()), (0, vec!["Reports go out on Fridays.".to_string()]));
    assert_eq!(agent.memory().len(), 2);
    assert!(agent.trace().iter().any(|e| e.kind == "note"));
    // An empty note is a bad reply; notes survive a restart.
    let Step::Ask(bad) = agent.reply(r#"{"tool":"remember","args":{"note":"  "}}"#) else { panic!() };
    assert!(bad.prompt.contains("the note is empty"));
    let cfg = Config { require_plan: false, ..Config::default() };
    let back = Agent::restore(cfg, tools(), &agent.snapshot()).unwrap();
    assert_eq!((back.memory().len(), back.new_notes().len()), (2, 1));
}

#[test]
fn only_the_newest_notes_are_shown_and_each_is_cut_short() {
    let notes: Vec<String> = (0..30).map(|i| format!("note number {i} {}", "z".repeat(400))).collect();
    let mut agent = Agent::new(Config { require_plan: false, ..Config::default() }, tools()).with_memory(notes);
    let Step::Ask(req) = agent.start("t", "") else { panic!() };
    assert!(!req.prompt.contains("note number 9 ") && req.prompt.contains("note number 10 ") && req.prompt.contains("note number 29 "));
    assert!(req.prompt.lines().filter(|l| l.starts_with("- ")).all(|l| l.len() <= 303));
}

#[test]
fn a_remembered_note_works_for_specialists_and_a_limited_lead() {
    let agent = Agent::new(Config { require_plan: false, ..Config::default() }, tools()).with_memory(vec![]).with_specialists(crew()).lead_tools(&["read"]);
    let mut agent = agent;
    let Step::Ask(req) = agent.start("t", "") else { panic!() };
    assert!(req.system.contains("remember") && req.system.contains("delegate") && !req.system.contains("\"tool\":\"save\""));
}

fn at(day: u64, hour: u64, minute: u64) -> u64 {
    // Day 0 is 1970-01-01, a Thursday; times are UTC.
    day * 86_400_000 + hour * 3_600_000 + minute * 60_000
}

#[test]
fn triggers_say_when_something_is_due() {
    use microagent::Trigger;
    let every = Trigger::Every { minutes: 30 };
    assert!(every.due(None, 0, 0) && !every.due(Some(at(5, 9, 0)), at(5, 9, 29), 0) && every.due(Some(at(5, 9, 0)), at(5, 9, 30), 0));
    // Daily at 09:00 (UTC): not before the slot, due after it, not again until tomorrow's.
    let daily = Trigger::Daily { hour: 9, minute: 0 };
    assert!(!daily.due(Some(at(5, 9, 5)), at(5, 12, 0), 0), "it already ran after today's slot");
    assert!(daily.due(Some(at(5, 8, 0)), at(5, 9, 0), 0), "the slot has come");
    assert!(!daily.due(Some(at(5, 8, 0)), at(5, 8, 59), 0), "not yet");
    assert!(daily.due(Some(at(5, 9, 5)), at(6, 9, 1), 0), "tomorrow's slot");
    // Created at 10:00, a daily 09:00 first runs tomorrow.
    assert!(!daily.due(Some(at(5, 10, 0)), at(5, 23, 0), 0) && daily.due(Some(at(5, 10, 0)), at(6, 9, 0), 0));
    // Time zones: 09:00 in Bangkok (UTC+7) is 02:00 UTC.
    assert!(daily.due(Some(at(5, 1, 0)), at(5, 2, 0), 420) && !daily.due(Some(at(5, 1, 0)), at(5, 1, 59), 420));
    // Weekly on Mondays (day 0): 1970-01-05 was a Monday (day 4).
    let weekly = Trigger::Weekly { day: 0, hour: 9, minute: 0 };
    assert!(weekly.due(Some(at(0, 0, 0)), at(4, 9, 0), 0));
    assert!(!weekly.due(Some(at(4, 9, 5)), at(10, 12, 0), 0), "Sunday: this week's Monday already ran");
    assert!(weekly.due(Some(at(4, 9, 5)), at(11, 9, 0), 0), "the next Monday");
    // Words and JSON.
    assert_eq!((every.describe().as_str(), daily.describe().as_str(), weekly.describe().as_str()), ("every 30 minutes", "daily at 09:00", "Mondays at 09:00"));
    assert_eq!(Trigger::Every { minutes: 120 }.describe(), "every 2 hours");
    for t in [every, daily, weekly] {
        assert_eq!(Trigger::from_json(&t.to_json()).unwrap(), t);
    }
    assert!(Trigger::from_json(r#"{"daily":[25,0]}"#).is_err() && Trigger::from_json(r#"{"every":0}"#).is_err() && Trigger::from_json("x").is_err());
}

#[test]
fn big_results_do_not_cause_a_summary_on_every_step() {
    // Each result is bigger than half the allowance: without a gap, every step would summarise.
    let cfg = Config { require_plan: false, summarize: true, history_chars: 1500, max_calls: 60, ..Config::default() };
    let mut agent = Agent::new(cfg, tools());
    let mut step = agent.start("t", "");
    let (mut asked, mut summaries) = (0, 0);
    for i in 0..18 {
        if let Step::Summarize(_) = step {
            summaries += 1;
            step = agent.summary("Short summary.");
        }
        let Step::Ask(_) = step else { panic!("{step:?}") };
        asked += 1;
        let Step::Run(call) = agent.reply(&format!(r#"{{"tool":"read","args":{{"name":"n{i}"}}}}"#)) else { panic!() };
        step = agent.result(call.id, ToolOutput::ok("x".repeat(1200)));
    }
    assert_eq!(asked, 18);
    assert!((1..=6).contains(&summaries), "{summaries} summaries in 18 steps");
    // The prompt stays small all the same: old results are trimmed.
    let Step::Ask(req) = (match step { Step::Summarize(_) => agent.summary("s"), s => s }) else { panic!() };
    assert!(req.prompt.len() < 4000, "{}", req.prompt.len());
}

#[test]
fn a_second_object_or_words_around_the_first_do_not_spoil_the_reply() {
    let mut agent = Agent::new(Config { require_plan: false, ..Config::default() }, tools());
    agent.start("t", "");
    // A small model that writes the arguments twice: the first complete object is used.
    let Step::Run(call) = agent.reply(r#"Sure! {"tool":"read","args":{"name":"a"}} {"name":"b"} hope that helps"#) else { panic!() };
    assert_eq!(call.args, json!({"name": "a"}));
    // Braces inside strings do not confuse it, and an object without an answer is skipped.
    let mut agent = Agent::new(Config { require_plan: false, ..Config::default() }, tools());
    agent.start("t", "");
    let Step::Done(o) = agent.reply(r#"{"note":"x"} then {"final":"a } brace and a { brace inside"}"#) else { panic!() };
    assert_eq!(o.summary, "a } brace and a { brace inside");
}

#[test]
fn a_reply_without_a_tool_is_told_the_three_shapes() {
    let mut agent = Agent::new(Config { require_plan: false, ..Config::default() }, tools());
    agent.start("t", "");
    let Step::Ask(req) = agent.reply(r#"{"args":{"cells":[]}}"#) else { panic!() };
    assert!(req.prompt.contains(r#"{"tool":"<name>","args":{...}}"#) && req.prompt.contains(r#"{"final":"what was done"}"#), "{}", req.prompt);
}

#[test]
fn arguments_written_beside_the_tool_name_are_still_taken() {
    let mut agent = Agent::new(Config { require_plan: false, ..Config::default() }, tools());
    agent.start("t", "");
    // Flattened: no `args` at all.
    let Step::Run(call) = agent.reply(r#"{"tool":"save","name":"a","text":"hello"}"#) else { panic!() };
    assert_eq!(call.args, json!({"name": "a", "text": "hello"}));
    agent.result(call.id, ToolOutput::ok("done"));
    // Split between the two, with `args` winning, and unknown keys dropped.
    let Step::Run(call) = agent.reply(r#"{"tool":"save","name":"x","thought":"hm","args":{"name":"b","text":"hi"}}"#) else { panic!() };
    assert_eq!(call.args, json!({"name": "b", "text": "hi"}));
}

#[test]
fn when_the_model_gives_up_the_failure_shows_what_it_last_sent() {
    let mut host = Script::new(&["no", "still no", "nope", "{\"tool\":\"nothing\"}"]);
    let out = run(&mut Agent::new(trusting(), tools()), &mut host, "t", "");
    assert!(matches!(&out.stop, Stop::Failed(w) if w.contains("kept giving") && w.contains("its last reply began: {\"tool\":\"nothing\"}")), "{:?}", out.stop);
}

fn skill(name: &str) -> microagent::Skill {
    microagent::Skill::new(name, &format!("Use when the work is {name}."), &format!("Rules for {name}: be exact."))
}

#[test]
fn skill_files_are_read_in_the_open_agent_skills_format() {
    use microagent::Skill;
    let s = Skill::parse("---\nname: slide-craft\ndescription: \"Use when making slides: one message per slide.\"\nlicense: MIT\nmetadata:\n  author: someone\n---\n\n# Slides\n\nOne message each.\n").unwrap();
    assert_eq!((s.name.as_str(), s.description.as_str()), ("slide-craft", "Use when making slides: one message per slide."));
    assert_eq!(s.body, "# Slides\n\nOne message each.");
    // A folded description, and Windows line endings.
    let f = Skill::parse("---\r\nname: x\r\ndescription: >\r\n  Use when a long\r\n  description is folded.\r\n---\r\nBody\r\n").unwrap();
    assert_eq!(f.description, "Use when a long description is folded.");
    assert_eq!(Skill::parse(&s.to_text()).unwrap(), s);
    for bad in ["no header", "---\nname: x\n---\nbody", "---\nname: Bad Name\ndescription: d\n---\nbody", "---\nname: ok\ndescription: d\n---\n", "---\nname: a--b\ndescription: d\n---\nbody", "---\nname: ok\ndescription: d"] {
        assert!(Skill::parse(bad).is_err(), "{bad:?}");
    }
    let long = format!("---\nname: big\ndescription: d\n---\n{}", "x".repeat(microagent::MAX_SKILL_CHARS + 1));
    assert!(Skill::parse(&long).unwrap_err().contains("at most"));
}

#[test]
fn the_model_sees_skill_names_and_loads_one_only_when_it_asks() {
    let mut agent = Agent::new(Config { require_plan: false, ..Config::default() }, tools()).with_skills(vec![skill("slide-craft"), skill("crisp-writing")]);
    let Step::Ask(req) = agent.start("t", "") else { panic!() };
    assert!(req.system.contains("use_skill") && req.system.contains("slide-craft: Use when the work is slide-craft."));
    assert!(!req.system.contains("Rules for slide-craft") && !req.prompt.contains("Rules for"), "the text is not sent until asked for");
    let Step::Ask(next) = agent.reply(r#"{"tool":"use_skill","args":{"name":"Slide-Craft"}}"#) else { panic!() };
    assert!(next.prompt.contains("Skill slide-craft:") && next.prompt.contains("Rules for slide-craft: be exact."));
    assert_eq!(agent.calls(), 0, "loading a skill is not a call");
    assert!(agent.trace().iter().any(|e| e.kind == "skill"));
    // Again: told it is loaded. Unknown: told the names.
    let Step::Ask(again) = agent.reply(r#"{"tool":"use_skill","args":{"name":"slide-craft"}}"#) else { panic!() };
    assert!(again.prompt.contains("already loaded"));
    let Step::Ask(unknown) = agent.reply(r#"{"tool":"use_skill","args":{"name":"nope"}}"#) else { panic!() };
    assert!(unknown.prompt.contains("no skill called nope") && unknown.prompt.contains("crisp-writing"));
    // The model may keep going with other tools afterwards.
    assert!(matches!(agent.reply(r#"{"tool":"read","args":{"name":"a"}}"#), Step::Run(_)));
}

#[test]
fn a_task_loads_at_most_six_skills_and_invalid_skills_are_left_out() {
    let all: Vec<_> = (0..8).map(|i| skill(&format!("skill-{i}"))).collect();
    let mut agent = Agent::new(Config { require_plan: false, ..Config::default() }, tools()).with_skills(all);
    agent.start("t", "");
    let mut last = String::new();
    for i in 0..7 {
        let Step::Ask(r) = agent.reply(&format!(r#"{{"tool":"use_skill","args":{{"name":"skill-{i}"}}}}"#)) else { panic!() };
        last = r.prompt;
    }
    assert!(last.contains("Enough skills are loaded"));
    let bad = microagent::Skill { name: "Bad Name".into(), description: "d".into(), body: "b".into() };
    let mut agent = Agent::new(Config::default(), tools()).with_skills(vec![bad]);
    let Step::Ask(req) = agent.start("t", "") else { panic!() };
    assert!(!req.system.contains("use_skill"), "no valid skill, no tool");
}

#[test]
fn loaded_skills_survive_a_restart() {
    let cfg = || Config { require_plan: false, ..Config::default() };
    let mut agent = Agent::new(cfg(), tools()).with_skills(vec![skill("crisp-writing")]);
    agent.start("t", "");
    agent.reply(r#"{"tool":"use_skill","args":{"name":"crisp-writing"}}"#);
    let mut back = Agent::restore(cfg(), tools(), &agent.snapshot()).unwrap().with_skills(vec![skill("crisp-writing")]);
    let Step::Ask(req) = back.resume() else { panic!() };
    assert!(req.prompt.contains("Skill crisp-writing:"));
    let Step::Ask(again) = back.reply(r#"{"tool":"use_skill","args":{"name":"crisp-writing"}}"#) else { panic!() };
    assert!(again.prompt.contains("already loaded"));
}
