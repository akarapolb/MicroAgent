# MicroAgent

A tiny agent loop you can embed in any app. **You bring the model and the tools. MicroAgent brings the plan, the approvals, the limits and a snapshot you can resume from.**

- **No I/O.** It never calls a model and never runs a tool. It tells you what to do next, and you tell it what happened. It fits a blocking program, async code, a GUI that waits for a click, a server, or WebAssembly.
- **One dependency** (`serde_json`). MIT licensed.
- **Any model.** The model only has to return text. It answers with one small JSON object per turn, so no function-calling API is needed.
- **Safe by default.** The model proposes a plan and you approve it once. Every tool has a risk class, and a policy says which ones need a yes before they run. Money always asks.

## The whole contract

Implement `Host` (three functions, two optional) and call `run`:

```rust
use microagent::{run, Agent, Config, Host, Kind, ModelRequest, Risk, Tool, ToolCall, ToolOutput};

struct MyApp;

impl Host for MyApp {
    // 1. Call your model. Send req.system and req.prompt; return its text.
    fn ask_model(&mut self, req: &ModelRequest) -> Result<String, String> { todo!() }

    // 2. Run a tool the model asked for. Return what happened, as text.
    fn run_tool(&mut self, call: &ToolCall) -> ToolOutput { todo!() }

    // 3. (optional) let a person approve the plan, and any call the policy says must ask.
    // fn approve_plan(&mut self, plan: &[String]) -> bool { true }
    // fn approve_call(&mut self, call: &ToolCall) -> bool { true }
}

let tools = vec![
    Tool::new("read_file", "Read a text file").arg("path", Kind::Text, "Where it is").risk(Risk::Read),
    Tool::new("write_file", "Write a text file").arg("path", Kind::Text, "Where").arg("text", Kind::Text, "Its text").risk(Risk::Write),
];
let mut agent = Agent::new(Config::default(), tools);
let outcome = run(&mut agent, &mut MyApp, "Copy notes.txt to backup.txt", "Files: notes.txt");
println!("{}: {}", outcome.stop.label(), outcome.summary);
```

That is everything. The rest is optional.

## Describing a tool

```rust
Tool::new("send_invoice", "Email an invoice to a customer")
    .arg("customer", Kind::Text, "Customer id")        // required
    .opt("note", Kind::Text, "A line for the email")   // optional
    .risk(Risk::Outside)                               // Read, Create, Write, Outside or Money
    .validate(|args| { /* your own limits; return the args or a sentence for the model */ Ok(args.clone()) })
```

MicroAgent checks the arguments against what you declared (right names, right types, numbers given as text become numbers, unknown extras are dropped) before your `run_tool` ever sees them. If the model gets them wrong, it is told what was wrong and tries again.

| Risk | Meaning | Cautious policy (default) |
| --- | --- | --- |
| `Read` | Looks, changes nothing | runs |
| `Create` | Makes something new, changes nothing that exists | runs |
| `Write` | Changes something in your app | asks |
| `Outside` | Acts outside your app (email, another service) | asks |
| `Money` | Moves money | **always asks** |

`Policy::TRUSTING` lets `Write` run too. Build your own `Policy` for anything else.

## Driving it yourself (async, UIs, other languages)

`Agent` is a state machine. Every method returns the next `Step`:

```rust
let mut step = agent.start("the task", "what the model should know first");
loop {
    step = match step {
        Step::Ask(req)   => match my_model(&req) { Ok(text) => agent.reply(&text), Err(e) => agent.fail(&e) },
        Step::Plan(plan) => agent.approve_plan(show_and_ask(&plan)),
        Step::Run(call)  => {
            let out = if call.ask && !person_says_yes(&call) { ToolOutput::declined() } else { run_my_tool(&call) };
            agent.result(call.id, out)
        }
        Step::Done(outcome) => break,
    };
}
```

Wait on a click, an HTTP call or a queue between any two steps. Calling a method at the wrong time is harmless: you get the step the agent is waiting on.

## Surviving a restart

```rust
let saved: String = agent.snapshot();                           // JSON text: keep it anywhere
let mut agent = Agent::restore(Config::default(), tools, &saved)?;
let step = agent.resume();                                      // exactly where it left off
```

## Verifying a result

When the model says it is done, check the work. If it is wrong, send it back:

```rust
if let Step::Done(o) = step {
    if let Some(problem) = my_check() { step = agent.reject_final(&problem); }
}
```

## Specialists and recipes

A **specialist** is a helper with its own tools and instructions. Give the agent some and the model can hand work over:

```rust
let crew = vec![
    Specialist::new("reader", "Looks things up").tools(&["read_file"]).instructions("Only look; never change anything."),
    Specialist::new("writer", "Saves files").tools(&["write_file"]),
];
let mut agent = Agent::new(Config::default(), tools).with_specialists(crew);
```

The model gets a `delegate` tool. `run` carries it out with a child agent that has only that specialist's tools, needs no plan of its own, and shares your host (model, tools, approvals); the child's summary comes back as the result. Driving it yourself, call `agent.start_child(&call)` for a `delegate` call and give the parent `agent.result(..)` when the child is done. Delegations run one at a time.

A **recipe** is a task you run again, with blanks to fill. It is plain data, so your app can store, edit and share them:

```rust
let r = Recipe::new("monthly", "Summarise spending for {month} and flag anything over {limit}")
    .param("month", "Which month").param("limit", "The limit").only_tools(&["read_file"]);
let task = r.fill(&[("month", "September"), ("limit", "50,000")])?;
let mut agent = Agent::new(Config::default(), r.tools_for(&all_tools));
```

## Memory and schedules

**Memory.** `Agent::new(..).with_memory(notes)` shows the model the notes you saved from earlier tasks and gives it a `remember` tool. The agent handles that tool itself (there is nothing for your app to run); when the task ends, store `agent.new_notes()` and pass them back next time. Only the newest 20 notes are shown, each cut to 300 characters.

**Triggers.** `Trigger::Every { minutes }`, `Daily { hour, minute }` and `Weekly { day, hour, minute }` say whether something is due: `trigger.due(last_run_ms, now_ms, utc_offset_minutes)`. It is plain arithmetic: your app owns the clock and decides what "due" does (run it, or only tell the person). Pass the time a schedule was made as its first `last_run_ms`, so a daily 09:00 created at 10:00 first fires tomorrow.

## Skills

A **skill** is an instruction pack in the open [Agent Skills](https://agentskills.io) format: a `SKILL.md` file with a `name` and a `description` in its header, then the instructions. Give the agent some and the model sees only each skill's name and description; it reads the full text with a `use_skill` tool when one fits the task, so a long guide costs nothing until it is used.

```rust
let skill = Skill::parse(&std::fs::read_to_string("slide-craft/SKILL.md")?)?;
let agent = Agent::new(Config::default(), tools).with_skills(vec![skill]);
```

The agent handles `use_skill` itself (there is nothing for your app to run), a task loads at most six, and loading is not counted against the call limit. Skills are plain text from whoever wrote them: read a skill before you add it, as you would any instructions you give the model.

## What it handles for you

- **A plan first**, approved once (`Config::require_plan`), with a step limit.
- **Retries** of unusable replies (fences, spoken thinking and `<think>` blocks are stripped), then a clear failure.
- **A call limit** (`max_calls`, default 12) and a stated reason for every stop (`Stop`).
- **"Done too early"** is caught once: if most of the plan is untouched, the model is sent back.
- **History** stays inside `history_chars`. Old turns are dropped whole so a call is never cut from its result, or, with `Config::summarize`, folded into a short summary the model writes (a `Step::Summarize`, which `Host::summarize` answers with an ordinary model call).
- **A trace** (`agent.trace()`) of plan, calls, results and stops, for showing progress or keeping a record.

## Which models work

MicroAgent asks the model for one JSON object per turn, so the model has to follow a format. What it was tried with (a plan, two or three tool calls, then done):

- **A strong model (a 27B model on vLLM with thinking off):** everything works, including specialists, memory and summaries.
- **A 7B model (qwen2.5:7b on Ollama):** finishes simple tasks in a few calls, and makes the occasional mistake. With specialists it invents tool names and runs out of calls, so keep specialists for stronger models.
- **A 3B model (qwen2.5:3b):** cannot keep to the format and fails after its retries.
- **A small reasoning model (qwen3:4b):** too slow to answer in time on that machine. Turn a model's thinking off where you can.

Unusable replies are retried with the reason (three times by default), and the first complete JSON object in a reply is used even when the model wrote words or a second object around it.

## What it deliberately does not do

It does not talk to any model or service, store anything, schedule anything, or run tools in parallel. Those belong to your app, where they fit your architecture. It also does not make a model clever: a weak model will still make weak plans, so the approvals and limits are there on purpose.

## Run the example

```sh
AI_URL=http://localhost:11434/v1/chat/completions AI_MODEL=qwen3:14b cargo run --example notes
```

The example uses any OpenAI-style server through `curl` and a folder of text files as its tools. Set `AI_KEY` if the server needs one.

## License

MIT. See `LICENSE`.
