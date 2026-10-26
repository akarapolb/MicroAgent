//! What the model may ask for, and what happens to each request.

use std::fmt;
use std::sync::Arc;

use serde_json::{json, Map, Value};

/// How much a tool can hurt, from safest to riskiest. The [`Policy`] says which ones need a yes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Risk {
    /// Looks at something and changes nothing.
    Read,
    /// Makes something new and changes nothing that exists (a new file, a new record).
    Create,
    /// Changes something inside your app.
    Write,
    /// Changes something outside your app (sends a message, calls another service).
    Outside,
    /// Moves money.
    Money,
}

impl Risk {
    fn label(self) -> &'static str {
        match self {
            Risk::Read => "read-only",
            Risk::Create => "makes new things",
            Risk::Write => "changes things",
            Risk::Outside => "acts outside the app",
            Risk::Money => "moves money",
        }
    }
}

/// Which risk classes need a yes from the person before they run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Policy {
    pub read: bool,
    pub create: bool,
    pub write: bool,
    pub outside: bool,
    /// Money always asks, whatever you put here.
    pub money: bool,
}

impl Policy {
    /// Ask before anything that changes something. Reads run at once.
    pub const CAUTIOUS: Policy = Policy { read: false, create: false, write: true, outside: true, money: true };
    /// Changes inside your app run at once; anything outside it, or money, still asks.
    pub const TRUSTING: Policy = Policy { read: false, create: false, write: false, outside: true, money: true };

    pub fn asks(&self, risk: Risk) -> bool {
        match risk {
            Risk::Read => self.read,
            Risk::Create => self.create,
            Risk::Write => self.write,
            Risk::Outside => self.outside,
            Risk::Money => true,
        }
    }
}

impl Default for Policy {
    fn default() -> Self {
        Policy::CAUTIOUS
    }
}

/// The type of one argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Text,
    Number,
    Bool,
    List,
    Object,
    Any,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Text => "text",
            Kind::Number => "number",
            Kind::Bool => "true or false",
            Kind::List => "list",
            Kind::Object => "object",
            Kind::Any => "any value",
        }
    }

    fn sample(self) -> Value {
        match self {
            Kind::Text => json!("..."),
            Kind::Number => json!(1),
            Kind::Bool => json!(true),
            Kind::List => json!([]),
            Kind::Object => json!({}),
            Kind::Any => json!("..."),
        }
    }
}

/// One argument of a tool.
#[derive(Clone, Debug)]
pub struct Arg {
    pub name: String,
    pub kind: Kind,
    pub required: bool,
    pub help: String,
}

/// A tool the model may call. Build one with [`Tool::new`] and chain the rest.
///
/// ```
/// use microagent::{Kind, Risk, Tool};
/// let t = Tool::new("read_file", "Read a text file")
///     .arg("path", Kind::Text, "Where the file is")
///     .risk(Risk::Read);
/// assert_eq!(t.name, "read_file");
/// ```
#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub help: String,
    pub risk: Risk,
    pub args: Vec<Arg>,
    /// An example call, shown to the model. Made from the arguments when you give none.
    pub example: Option<Value>,
    validator: Option<Validator>,
    summarizer: Option<Summarizer>,
}

type Validator = Arc<dyn Fn(&Value) -> Result<Value, String> + Send + Sync>;
type Summarizer = Arc<dyn Fn(&Value) -> String + Send + Sync>;

impl fmt::Debug for Tool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tool").field("name", &self.name).field("risk", &self.risk).field("args", &self.args).finish()
    }
}

impl Tool {
    pub fn new(name: &str, help: &str) -> Tool {
        Tool { name: name.to_owned(), help: help.to_owned(), risk: Risk::Read, args: Vec::new(), example: None, validator: None, summarizer: None }
    }

    pub fn risk(mut self, risk: Risk) -> Tool {
        self.risk = risk;
        self
    }

    /// A required argument.
    pub fn arg(mut self, name: &str, kind: Kind, help: &str) -> Tool {
        self.args.push(Arg { name: name.to_owned(), kind, required: true, help: help.to_owned() });
        self
    }

    /// An argument the model may leave out.
    pub fn opt(mut self, name: &str, kind: Kind, help: &str) -> Tool {
        self.args.push(Arg { name: name.to_owned(), kind, required: false, help: help.to_owned() });
        self
    }

    /// How a call to this tool reads to a person, for an approval card or a progress line:
    /// "File a purchase request for 250,000 baht" rather than the tool name and its arguments.
    /// Put on [`ToolCall::summary`]; without one it is the name and the arguments.
    pub fn summary(mut self, describe: impl Fn(&Value) -> String + Send + Sync + 'static) -> Tool {
        self.summarizer = Some(Arc::new(describe));
        self
    }

    pub(crate) fn summary_of(&self, args: &Value) -> String {
        match &self.summarizer {
            Some(f) => f(args),
            None => format!("{} {args}", self.name),
        }
    }

    pub fn example(mut self, example: Value) -> Tool {
        self.example = Some(example);
        self
    }

    /// Your own check of the arguments, run after the declared ones pass. Return the arguments
    /// (changed if you like) or a sentence the model can act on. Keeps limits such as "at most
    /// 200 cells" next to the tool instead of inside your host.
    pub fn validate(mut self, check: impl Fn(&Value) -> Result<Value, String> + Send + Sync + 'static) -> Tool {
        self.validator = Some(Arc::new(check));
        self
    }

    /// The line the model reads about this tool.
    pub(crate) fn describe(&self) -> String {
        let example = self.example.clone().unwrap_or_else(|| {
            let mut args = Map::new();
            for arg in self.args.iter().filter(|a| a.required) {
                args.insert(arg.name.clone(), arg.kind.sample());
            }
            json!({"tool": self.name, "args": Value::Object(args)})
        });
        let mut s = format!("  {example}  {} [{}]", self.help, self.risk.label());
        for a in &self.args {
            s.push_str(&format!(
                "\n      {}{}: {}. {}",
                a.name,
                if a.required { "" } else { " (optional)" },
                a.kind.name(),
                a.help
            ));
        }
        s
    }

    /// Checks the model's arguments against the declared ones and returns them in one clean
    /// shape: only the declared names, numbers as numbers, text as text.
    pub fn check(&self, given: &Value) -> Result<Value, String> {
        let empty = Map::new();
        let obj = match given {
            Value::Object(o) => o,
            Value::Null => &empty,
            _ => return Err("args must be an object".into()),
        };
        let mut out = Map::new();
        for a in &self.args {
            match obj.get(&a.name) {
                None | Some(Value::Null) if a.required => {
                    return Err(format!("args.{} is missing ({}: {})", a.name, a.kind.name(), a.help))
                }
                None | Some(Value::Null) => {}
                Some(v) => {
                    out.insert(a.name.clone(), coerce(a, v)?);
                }
            }
        }
        let clean = Value::Object(out);
        match &self.validator {
            Some(check) => check(&clean),
            None => Ok(clean),
        }
    }
}

fn coerce(a: &Arg, v: &Value) -> Result<Value, String> {
    let bad = || format!("args.{} must be {}", a.name, a.kind.name());
    Ok(match (a.kind, v) {
        (Kind::Any, v) => v.clone(),
        (Kind::Text, Value::String(s)) => json!(s),
        (Kind::Text, Value::Number(n)) => json!(n.to_string()),
        (Kind::Number, Value::Number(_)) => v.clone(),
        (Kind::Number, Value::String(s)) => {
            let n: f64 = s.trim().parse().map_err(|_| bad())?;
            if n.fract() == 0.0 && n.abs() < 9e15 { json!(n as i64) } else { json!(n) }
        }
        (Kind::Bool, Value::Bool(_)) => v.clone(),
        (Kind::List, Value::Array(_)) => v.clone(),
        (Kind::Object, Value::Object(_)) => v.clone(),
        _ => return Err(bad()),
    })
}

/// A request from the model to run one tool. Your app runs it and gives back a [`ToolOutput`].
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    /// Hand this back with the result.
    pub id: u32,
    pub name: String,
    /// The arguments, already checked against the tool's declaration.
    pub args: Value,
    /// The risk class of the tool.
    pub risk: Risk,
    /// True when the policy says the person must say yes first.
    pub ask: bool,
    /// The call in words, for showing to a person: from [`Tool::summary`], else the name and the
    /// arguments.
    pub summary: String,
}

/// What a tool gave back, as text the model reads.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolOutput {
    pub text: String,
    pub ok: bool,
}

impl ToolOutput {
    pub fn ok(text: impl Into<String>) -> ToolOutput {
        ToolOutput { text: text.into(), ok: true }
    }

    /// The tool could not do it. The model is told why and can try something else.
    pub fn error(text: impl Into<String>) -> ToolOutput {
        ToolOutput { text: text.into(), ok: false }
    }

    /// The person said no.
    pub fn declined() -> ToolOutput {
        ToolOutput { text: "The person declined this action. Do not try it again; finish or choose another way.".into(), ok: false }
    }

    pub(crate) fn for_model(&self) -> String {
        if self.ok || self.text.starts_with("The person declined") {
            self.text.clone()
        } else {
            format!("Failed: {}", self.text)
        }
    }
}
