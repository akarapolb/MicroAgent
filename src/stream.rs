//! Pulling the text of a final answer out of a reply while it is still arriving.

/// Pulls the text of `{"final": "..."}` out of a reply while it streams in, decoding JSON
/// escapes as they complete. Anything else (a tool call, words before the object, another
/// key first) yields nothing, and the answer arrives whole at the end as before.
#[derive(Default)]
pub struct FinalStream {
    buf: String,
    /// 0 = still reading the opening, 1 = inside the string, 2 = finished or not a final.
    state: u8,
    /// Byte offset in `buf` of the first string character not yet decoded.
    pos: usize,
}

impl FinalStream {
    pub fn feed(&mut self, chunk: &str) -> String {
        const OPENING: &[u8] = br#"{"final":""#;
        self.buf.push_str(chunk);
        let mut out = String::new();
        if self.state == 0 {
            let mut matched = 0;
            let mut end = None;
            for (i, c) in self.buf.char_indices() {
                if c.is_whitespace() {
                    continue;
                }
                if c as u32 == OPENING[matched] as u32 {
                    matched += 1;
                    if matched == OPENING.len() {
                        end = Some(i + 1);
                        break;
                    }
                } else {
                    self.state = 2;
                    return out;
                }
            }
            match end {
                Some(e) => {
                    self.state = 1;
                    self.pos = e;
                }
                None => return out,
            }
        }
        if self.state == 1 {
            let rest = &self.buf[self.pos..];
            let mut used = 0;
            let mut it = rest.char_indices().peekable();
            while let Some((i, c)) = it.next() {
                match c {
                    '"' => {
                        self.state = 2;
                        used = i + 1;
                        break;
                    }
                    '\\' => {
                        let Some(&(_, e)) = it.peek() else { break }; // wait for the escape
                        let decoded = match e {
                            'n' => Some('\n'),
                            't' => Some('\t'),
                            'r' => Some('\r'),
                            '"' | '\\' | '/' => Some(e),
                            'b' | 'f' => None,
                            'u' => {
                                let hex = rest.get(i + 2..i + 6);
                                match hex {
                                    Some(h) if h.chars().all(|x| x.is_ascii_hexdigit()) => {
                                        it.next();
                                        for _ in 0..4 {
                                            it.next();
                                        }
                                        if let Some(ch) = u32::from_str_radix(h, 16).ok().and_then(char::from_u32) {
                                            out.push(ch);
                                        }
                                        used = i + 6;
                                        continue;
                                    }
                                    Some(_) => None,
                                    None => break, // wait for all four digits
                                }
                            }
                            _ => Some(e),
                        };
                        it.next();
                        if let Some(d) = decoded {
                            out.push(d);
                        }
                        used = i + 1 + e.len_utf8();
                    }
                    _ => {
                        out.push(c);
                        used = i + c.len_utf8();
                    }
                }
            }
            self.pos += used;
        }
        out
    }
}


#[cfg(test)]
mod tests {
    use super::FinalStream;

    #[test]
    fn a_final_answer_streams_as_it_is_written() {
        // Fed in awkward pieces: the opening split, an escape split, a \u escape, non-ASCII.
        let mut d = FinalStream::default();
        let pieces = ["  {\"fi", "nal\": \"Total ", "is 4,200 [1].\\", "n caf\u{e9} \\u0e01", "\" , \"x\": {}}"];
        let got: String = pieces.iter().map(|p| d.feed(p)).collect();
        assert_eq!(got, "Total is 4,200 [1].\n caf\u{e9} \u{e01}");
        assert_eq!(d.feed(" more"), "");
        // A tool call, or words before the object, stream nothing.
        let mut d = FinalStream::default();
        assert_eq!(d.feed("{\"tool\":\"x\"}"), "");
        let mut d = FinalStream::default();
        assert_eq!(d.feed("Sure. {\"final\":\"x\"}"), "");
        // Escaped quotes stay in the text.
        let mut d = FinalStream::default();
        assert_eq!(d.feed(r#"{"final":"the \"A\" plan"}"#), "the \"A\" plan");
    }
}
