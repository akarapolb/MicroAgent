//! Skills: instruction packs the model loads only when they fit the task.
//!
//! The format is the open Agent Skills one (a `SKILL.md` file): a short header with a `name` and
//! a `description`, then the instructions. The model is shown every skill's name and description
//! and calls `use_skill` to read the full text of the one it needs, so a long guide costs nothing
//! until it is used.

/// One skill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skill {
    /// Lowercase letters, digits and single hyphens, up to 64 characters.
    pub name: String,
    /// When to use it, up to 1,024 characters. This is what the model reads to decide.
    pub description: String,
    /// The instructions.
    pub body: String,
}

/// The longest skill text accepted.
pub const MAX_SKILL_CHARS: usize = 24_000;

impl Skill {
    pub fn new(name: &str, description: &str, body: &str) -> Skill {
        Skill { name: name.trim().to_owned(), description: description.trim().to_owned(), body: body.trim().to_owned() }
    }

    /// Reads a skill from `SKILL.md` text: a header between two `---` lines with `name` and
    /// `description` (other keys, such as `license`, are ignored), then the instructions.
    ///
    /// ```
    /// use microagent::Skill;
    /// let s = Skill::parse("---\nname: crisp-writing\ndescription: Use when writing for executives.\n---\nShort sentences.").unwrap();
    /// assert_eq!(s.name, "crisp-writing");
    /// assert_eq!(s.body, "Short sentences.");
    /// ```
    pub fn parse(text: &str) -> Result<Skill, String> {
        let text = text.trim_start_matches('\u{feff}').replace("\r\n", "\n");
        let rest = text
            .strip_prefix("---\n")
            .ok_or("a skill file starts with a --- line, then name: and description:, then another --- line")?;
        let end = rest.find("\n---").ok_or("the header of a skill file must end with a --- line")?;
        let (header, after) = rest.split_at(end);
        let body = after[4..].trim_start_matches(['\n', ' ']).trim_end();
        let (mut name, mut description) = (String::new(), String::new());
        let mut key: Option<&str> = None;
        for line in header.lines() {
            if line.starts_with([' ', '\t']) {
                // A continuation of a folded value.
                if let Some(k) = key {
                    let add = line.trim();
                    match k {
                        "name" => {
                            name.push(' ');
                            name.push_str(add);
                        }
                        "description" => {
                            description.push(' ');
                            description.push_str(add);
                        }
                        _ => {}
                    }
                }
            } else if let Some((k, v)) = line.split_once(':') {
                let v = v.trim().trim_matches(['"', '\'']).trim_start_matches(['>', '|']).trim();
                key = Some(match k.trim() {
                    "name" => "name",
                    "description" => "description",
                    _ => "other",
                });
                match key {
                    Some("name") => name = v.to_owned(),
                    Some("description") => description = v.to_owned(),
                    _ => {}
                }
            }
        }
        let skill = Skill::new(&name, &description, body);
        skill.check()?;
        Ok(skill)
    }

    /// Whether the skill is acceptable: a proper name, a description, instructions, and a size limit.
    pub fn check(&self) -> Result<(), String> {
        let ok_name = !self.name.is_empty()
            && self.name.len() <= 64
            && !self.name.starts_with('-')
            && !self.name.ends_with('-')
            && !self.name.contains("--")
            && self.name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-');
        if !ok_name {
            return Err("the name of a skill is lowercase letters, digits and hyphens, up to 64 characters (like crisp-writing)".into());
        }
        if self.description.is_empty() || self.description.chars().count() > 1024 {
            return Err("a skill needs a description of up to 1,024 characters saying when to use it".into());
        }
        if self.body.is_empty() {
            return Err("a skill needs instructions after its header".into());
        }
        if self.body.chars().count() > MAX_SKILL_CHARS {
            return Err(format!("a skill is at most {MAX_SKILL_CHARS} characters"));
        }
        Ok(())
    }

    /// The skill as `SKILL.md` text.
    pub fn to_text(&self) -> String {
        format!("---\nname: {}\ndescription: {}\n---\n\n{}\n", self.name, self.description.replace('\n', " "), self.body)
    }
}
