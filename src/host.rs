//! The easy way in: implement [`Host`] and call [`run`]. If you need async, a UI that waits for
//! clicks, or another language, skip this file and drive [`crate::Agent`] step by step.

use crate::agent::{Agent, Event, ModelRequest, Outcome, Step, Stop, DELEGATE};
use crate::tool::{ToolCall, ToolOutput};

/// Your app, as the agent sees it. Three things to write.
pub trait Host {
    /// Send the request to your model and return its text. An `Err` ends the task.
    fn ask_model(&mut self, request: &ModelRequest) -> Result<String, String>;

    /// Run a tool. Checking the arguments' meaning is your job here: if they make no sense,
    /// return [`ToolOutput::error`] with a sentence the model can act on.
    fn run_tool(&mut self, call: &ToolCall) -> ToolOutput;

    /// Write a summary of older steps (only asked when `Config::summarize` is on). It is an
    /// ordinary model request, so the default simply uses [`Host::ask_model`].
    fn summarize(&mut self, request: &ModelRequest) -> Result<String, String> {
        self.ask_model(request)
    }

    /// Show the plan to the person. Return false to end the task. Defaults to yes.
    fn approve_plan(&mut self, _plan: &[String]) -> bool {
        true
    }

    /// A tool the policy says needs a yes. Return false to refuse. Defaults to yes.
    fn approve_call(&mut self, _call: &ToolCall) -> bool {
        true
    }

    /// Called for each trace event as it happens. Optional.
    fn event(&mut self, _event: &Event) {}
}

/// Runs a whole task with a [`Host`]. Blocks until it ends. A `delegate` call is handed to a
/// child agent for that specialist, which shares your host (its model, tools and approvals).
pub fn run(agent: &mut Agent, host: &mut impl Host, task: &str, context: &str) -> Outcome {
    let first = agent.start(task, context);
    drive(agent, host, first)
}

fn drive(agent: &mut Agent, host: &mut impl Host, first: Step) -> Outcome {
    let mut seen = 0;
    let mut step = first;
    loop {
        for e in &agent.trace()[seen..] {
            host.event(e);
        }
        seen = agent.trace().len();
        step = match step {
            Step::Ask(req) => match host.ask_model(&req) {
                Ok(text) => agent.reply(&text),
                Err(why) => agent.fail(&why),
            },
            Step::Summarize(req) => match host.summarize(&req) {
                Ok(text) => agent.summary(&text),
                Err(_) => agent.summary_failed(),
            },
            Step::Plan(plan) => {
                let yes = host.approve_plan(&plan);
                agent.approve_plan(yes)
            }
            Step::Run(call) if call.name == DELEGATE => {
                let out = match agent.start_child(&call) {
                    Ok((mut child, first)) => {
                        let outcome = drive(&mut child, host, first);
                        match outcome.stop {
                            Stop::Done => ToolOutput::ok(outcome.summary),
                            other => ToolOutput::error(format!("the specialist {}", other.label())),
                        }
                    }
                    Err(why) => ToolOutput::error(why),
                };
                agent.result(call.id, out)
            }
            Step::Run(call) => {
                let out = if call.ask && !host.approve_call(&call) { ToolOutput::declined() } else { host.run_tool(&call) };
                agent.result(call.id, out)
            }
            Step::Done(outcome) => {
                for e in &agent.trace()[seen..] {
                    host.event(e);
                }
                return outcome;
            }
        };
    }
}
