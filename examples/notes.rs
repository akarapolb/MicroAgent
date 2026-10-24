//! A complete MicroAgent host in one file: a folder of notes as the tools, and any OpenAI-style
//! server as the model (through `curl`, so the example needs no other dependency).
//!
//!     AI_URL=http://localhost:11434/v1/chat/completions AI_MODEL=qwen3:14b \
//!         cargo run -p microagent --example notes -- "Copy a.txt to b.txt"
//!
//! `AI_KEY` is sent as a bearer token if set; `AI_EXTRA` adds fields to the request body. The notes live in `./notes-demo/`.

use microagent::{run, Agent, Config, Host, Kind, ModelRequest, Policy, Risk, Tool, ToolCall, ToolOutput};
use serde_json::{json, Value};
use std::{fs, io::Write as _, path::PathBuf, process::Command};

struct Notes {
    dir: PathBuf,
    url: String,
    model: String,
    key: String,
}

impl Notes {
    fn path(&self, name: &str) -> Result<PathBuf, String> {
        if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') {
            return Err("a note name is a plain file name".into());
        }
        Ok(self.dir.join(name))
    }
}

impl Host for Notes {
    // 1. The model: POST the request and return the text of its answer.
    fn ask_model(&mut self, req: &ModelRequest) -> Result<String, String> {
        let mut body = json!({
            "model": self.model,
            "max_tokens": req.max_tokens,
            "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": req.prompt}],
        });
        // AI_EXTRA: a JSON object of extra fields, for example to turn a model's thinking off.
        if let Ok(Value::Object(extra)) = serde_json::from_str(&std::env::var("AI_EXTRA").unwrap_or_default()) {
            for (k, v) in extra {
                body[k] = v;
            }
        }
        let mut cmd = Command::new("curl");
        cmd.args(["-sS", "-m", "120", "-H", "content-type: application/json"]);
        if !self.key.is_empty() {
            cmd.args(["-H", &format!("authorization: Bearer {}", self.key)]);
        }
        let out = cmd.args(["-d", &body.to_string(), &self.url]).output().map_err(|e| e.to_string())?;
        let v: Value = serde_json::from_slice(&out.stdout).map_err(|_| "the server did not answer with JSON".to_string())?;
        v["choices"][0]["message"]["content"].as_str().map(str::to_owned).ok_or_else(|| v.to_string())
    }

    // 2. The tools: do what was asked, say what happened.
    fn run_tool(&mut self, call: &ToolCall) -> ToolOutput {
        let arg = |k: &str| call.args[k].as_str().unwrap_or("").to_owned();
        match call.name.as_str() {
            "list_notes" => match fs::read_dir(&self.dir) {
                Ok(d) => ToolOutput::ok(d.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect::<Vec<_>>().join("\n")),
                Err(e) => ToolOutput::error(e.to_string()),
            },
            "read_note" => match self.path(&arg("name")).and_then(|p| fs::read_to_string(p).map_err(|e| e.to_string())) {
                Ok(t) => ToolOutput::ok(t),
                Err(e) => ToolOutput::error(e),
            },
            "write_note" => match self.path(&arg("name")).and_then(|p| fs::write(p, arg("text")).map_err(|e| e.to_string())) {
                Ok(()) => ToolOutput::ok("saved"),
                Err(e) => ToolOutput::error(e),
            },
            other => ToolOutput::error(format!("{other} is not a tool")),
        }
    }

    // 3. The person: show the plan, and ask before a change.
    fn approve_plan(&mut self, plan: &[String]) -> bool {
        println!("Plan:");
        for (i, s) in plan.iter().enumerate() {
            println!("  {}. {s}", i + 1);
        }
        ask("Go ahead?")
    }

    fn approve_call(&mut self, call: &ToolCall) -> bool {
        ask(&format!("Allow {} {}?", call.name, call.args))
    }

    fn event(&mut self, e: &microagent::Event) {
        println!("  [{}] {}", e.kind, e.text.chars().take(100).collect::<String>());
    }
}

fn ask(question: &str) -> bool {
    print!("{question} [y/N] ");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).ok();
    line.trim().eq_ignore_ascii_case("y")
}

fn main() {
    let task = std::env::args().nth(1).unwrap_or_else(|| "List the notes and say what is in them.".into());
    let dir = PathBuf::from("notes-demo");
    fs::create_dir_all(&dir).ok();
    if fs::read_dir(&dir).map(|d| d.count()).unwrap_or(0) == 0 {
        fs::write(dir.join("a.txt"), "Buy milk.").ok();
    }
    let mut host = Notes {
        dir,
        url: std::env::var("AI_URL").unwrap_or_else(|_| "http://localhost:11434/v1/chat/completions".into()),
        model: std::env::var("AI_MODEL").unwrap_or_else(|_| "qwen3:14b".into()),
        key: std::env::var("AI_KEY").unwrap_or_default(),
    };
    let tools = vec![
        Tool::new("list_notes", "List the notes").risk(Risk::Read),
        Tool::new("read_note", "Read a note").arg("name", Kind::Text, "File name, like a.txt").risk(Risk::Read),
        Tool::new("write_note", "Write a note, replacing it").arg("name", Kind::Text, "File name").arg("text", Kind::Text, "Its text").risk(Risk::Write),
    ];
    let cfg = Config { policy: Policy::CAUTIOUS, instructions: "You help with a folder of plain text notes.".into(), ..Config::default() };
    let outcome = run(&mut Agent::new(cfg, tools), &mut host, &task, "");
    println!("\n{}: {}", outcome.stop.label(), outcome.summary);
}
