//! # MicroAgent
//!
//! A tiny agent loop to embed in any app. You bring the model and the tools; MicroAgent
//! brings the plan, the approvals, the limits and a snapshot you can resume from.
//!
//! It does **no I/O**. It never calls a model and never runs a tool: it tells you what to
//! do next and you tell it what happened. So it fits any architecture: blocking, async,
//! a GUI that waits for a click, a server, WebAssembly.
//!
//! ## The whole contract
//!
//! Implement [`Host`] (three functions) and call [`run`]:
//!
//! ```
//! use microagent::{run, Agent, Config, Host, Kind, ModelRequest, Risk, Tool, ToolCall, ToolOutput};
//!
//! struct MyApp;
//! impl Host for MyApp {
//!     // 1. Call your model. Any model: it only has to return text.
//!     fn ask_model(&mut self, _req: &ModelRequest) -> Result<String, String> {
//!         Ok(r#"{"final":"hello"}"#.into()) // a real app sends req.system and req.prompt
//!     }
//!     // 2. Run a tool the model asked for.
//!     fn run_tool(&mut self, call: &ToolCall) -> ToolOutput {
//!         ToolOutput::ok(format!("ran {}", call.name))
//!     }
//!     // 3. (optional) approve_plan and approve_call let a person say yes or no.
//! }
//!
//! let tools = vec![Tool::new("lookup", "Look something up").arg("q", Kind::Text, "What to find").risk(Risk::Read)];
//! let mut agent = Agent::new(Config { require_plan: false, ..Config::default() }, tools);
//! let outcome = run(&mut agent, &mut MyApp, "Say hello", "");
//! assert_eq!(outcome.summary, "hello");
//! ```
//!
//! ## Driving it yourself
//!
//! [`Agent`] is a state machine. `start`, `reply`, `approve_plan` and `result` each return
//! the next [`Step`]: ask the model, show a plan, run a tool, or done. [`Agent::snapshot`]
//! and [`Agent::restore`] let a task survive a restart.
//!
//! MicroAgent is MIT licensed and has one dependency, `serde_json`.

mod agent;
mod host;
mod recipe;
mod schedule;
mod skill;
mod tool;

pub use agent::{Agent, Config, Event, ModelRequest, Outcome, Step, Stop, DELEGATE, REMEMBER, USE_SKILL};
pub use host::{run, Host};
pub use recipe::{Param, Recipe, Specialist};
pub use schedule::Trigger;
pub use skill::{Skill, MAX_SKILL_CHARS};
pub use tool::{Arg, Kind, Policy, Risk, Tool, ToolCall, ToolOutput};
