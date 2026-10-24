//! Specialists and recipes: two small ways to reuse what you have already set up.
//!
//! * A [`Specialist`] is a named helper with its own tools and instructions. Give a parent
//!   [`Agent`](crate::Agent) some with [`Agent::with_specialists`](crate::Agent::with_specialists)
//!   and the model gets a `delegate` tool; [`run`](crate::run) carries the delegation out with
//!   a child agent and hands the child's summary back.
//! * A [`Recipe`] is a task you run again and again, with blanks to fill in. It is plain data
//!   (`to_json` / `from_json`), so your app can store and share them.

use serde_json::{json, Value};

use crate::tool::Tool;

/// A named helper for one kind of work.
#[derive(Clone, Debug)]
pub struct Specialist {
    pub name: String,
    /// What it is good at, shown to the parent model so it knows when to hand work over.
    pub help: String,
    /// The names of the tools it may use (the parent's registered tools, by name).
    pub tools: Vec<String>,
    /// Extra words for the child model: how this specialist should work.
    pub instructions: String,
}

impl Specialist {
    pub fn new(name: &str, help: &str) -> Specialist {
        Specialist { name: name.to_owned(), help: help.to_owned(), tools: Vec::new(), instructions: String::new() }
    }

    /// The tools it may use, by name.
    pub fn tools(mut self, names: &[&str]) -> Specialist {
        self.tools = names.iter().map(|n| (*n).to_owned()).collect();
        self
    }

    pub fn instructions(mut self, text: &str) -> Specialist {
        self.instructions = text.to_owned();
        self
    }
}

/// A blank in a recipe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub help: String,
}

/// A task you run again and again. Write blanks as `{name}` in the task text.
///
/// ```
/// use microagent::Recipe;
/// let r = Recipe::new("monthly", "Summarise spending for {month} and flag anything over {limit}")
///     .param("month", "Which month")
///     .param("limit", "The spending limit");
/// let task = r.fill(&[("month", "September"), ("limit", "50,000")]).unwrap();
/// assert_eq!(task, "Summarise spending for September and flag anything over 50,000");
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recipe {
    pub name: String,
    pub task: String,
    pub params: Vec<Param>,
    /// The tools this recipe may use, by name. Empty means all of them.
    pub tools: Vec<String>,
}

impl Recipe {
    pub fn new(name: &str, task: &str) -> Recipe {
        Recipe { name: name.to_owned(), task: task.to_owned(), params: Vec::new(), tools: Vec::new() }
    }

    pub fn param(mut self, name: &str, help: &str) -> Recipe {
        self.params.push(Param { name: name.to_owned(), help: help.to_owned() });
        self
    }

    /// Limits the recipe to these tools, by name.
    pub fn only_tools(mut self, names: &[&str]) -> Recipe {
        self.tools = names.iter().map(|n| (*n).to_owned()).collect();
        self
    }

    /// The names written as `{name}` in the task text, in order, each once.
    pub fn blanks(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut rest = self.task.as_str();
        while let Some(a) = rest.find('{') {
            let Some(b) = rest[a..].find('}') else { break };
            let name = &rest[a + 1..a + b];
            if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') && !out.iter().any(|n| n == name) {
                out.push(name.to_owned());
            }
            rest = &rest[a + b + 1..];
        }
        out
    }

    /// The task with every blank filled in. A blank with no value is an error that names it.
    pub fn fill(&self, values: &[(&str, &str)]) -> Result<String, String> {
        let mut task = self.task.clone();
        for name in self.blanks() {
            let value = values.iter().find(|(k, _)| *k == name).map(|(_, v)| v.trim()).filter(|v| !v.is_empty()).ok_or_else(|| format!("{name} needs a value"))?;
            task = task.replace(&format!("{{{name}}}"), value);
        }
        Ok(task)
    }

    /// Keeps only the tools this recipe may use.
    pub fn tools_for(&self, all: &[Tool]) -> Vec<Tool> {
        if self.tools.is_empty() {
            return all.to_vec();
        }
        all.iter().filter(|t| self.tools.contains(&t.name)).cloned().collect()
    }

    pub fn to_json(&self) -> String {
        json!({
            "name": self.name,
            "task": self.task,
            "params": self.params.iter().map(|p| json!({"name": p.name, "help": p.help})).collect::<Vec<_>>(),
            "tools": self.tools,
        })
        .to_string()
    }

    pub fn from_json(text: &str) -> Result<Recipe, String> {
        let v: Value = serde_json::from_str(text).map_err(|_| "a recipe is JSON".to_string())?;
        let name = v["name"].as_str().filter(|n| !n.trim().is_empty()).ok_or("a recipe needs a name")?;
        let task = v["task"].as_str().filter(|n| !n.trim().is_empty()).ok_or("a recipe needs a task")?;
        let strings = |k: &str| -> Vec<String> { v[k].as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect()).unwrap_or_default() };
        let params = v["params"]
            .as_array()
            .map(|a| a.iter().filter_map(|p| Some(Param { name: p["name"].as_str()?.to_owned(), help: p["help"].as_str().unwrap_or("").to_owned() })).collect())
            .unwrap_or_default();
        Ok(Recipe { name: name.trim().to_owned(), task: task.to_owned(), params, tools: strings("tools") })
    }
}
