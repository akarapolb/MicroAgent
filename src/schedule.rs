//! When a recipe is due. Pure arithmetic on timestamps: your app owns the clock and the
//! loop that asks "is anything due?", and decides what to do when it is (run it, or just tell
//! the person).

use serde_json::{json, Value};

const MINUTE: u64 = 60_000;
const DAY: u64 = 24 * 60 * MINUTE;

/// How often something is due.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trigger {
    /// Every this many minutes.
    Every { minutes: u32 },
    /// Once a day at this local time.
    Daily { hour: u8, minute: u8 },
    /// Once a week on this local day (0 is Monday) and time.
    Weekly { day: u8, hour: u8, minute: u8 },
}

impl Trigger {
    /// Whether it is due at `now_ms`, given when it last ran (milliseconds since the Unix epoch).
    /// `utc_offset_minutes` is the local time zone (for example 420 for Bangkok). For a schedule
    /// that has never run, pass the time it was created as `last_run_ms`, so a daily 09:00 made
    /// at 10:00 first runs tomorrow. `None` means it has never run and is due at once.
    pub fn due(&self, last_run_ms: Option<u64>, now_ms: u64, utc_offset_minutes: i32) -> bool {
        let Some(last) = last_run_ms else { return true };
        match *self {
            Trigger::Every { minutes } => now_ms.saturating_sub(last) >= u64::from(minutes.max(1)) * MINUTE,
            Trigger::Daily { hour, minute } => last < latest_slot(now_ms, utc_offset_minutes, None, hour, minute),
            Trigger::Weekly { day, hour, minute } => last < latest_slot(now_ms, utc_offset_minutes, Some(day % 7), hour, minute),
        }
    }

    /// A short phrase for showing it: `every 30 minutes`, `daily at 09:00`, `Mondays at 09:00`.
    pub fn describe(&self) -> String {
        match *self {
            Trigger::Every { minutes } if minutes % 60 == 0 && minutes > 0 => format!("every {} hour{}", minutes / 60, if minutes == 60 { "" } else { "s" }),
            Trigger::Every { minutes } => format!("every {minutes} minute{}", if minutes == 1 { "" } else { "s" }),
            Trigger::Daily { hour, minute } => format!("daily at {hour:02}:{minute:02}"),
            Trigger::Weekly { day, hour, minute } => {
                let d = ["Mondays", "Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays", "Sundays"][usize::from(day % 7)];
                format!("{d} at {hour:02}:{minute:02}")
            }
        }
    }

    pub fn to_json(&self) -> String {
        match *self {
            Trigger::Every { minutes } => json!({"every": minutes}),
            Trigger::Daily { hour, minute } => json!({"daily": [hour, minute]}),
            Trigger::Weekly { day, hour, minute } => json!({"weekly": [day, hour, minute]}),
        }
        .to_string()
    }

    pub fn from_json(text: &str) -> Result<Trigger, String> {
        let v: Value = serde_json::from_str(text).map_err(|_| "a trigger is JSON".to_string())?;
        let n = |x: &Value| x.as_u64().and_then(|n| u8::try_from(n).ok());
        if let Some(m) = v["every"].as_u64().and_then(|m| u32::try_from(m).ok()).filter(|m| *m >= 1) {
            return Ok(Trigger::Every { minutes: m });
        }
        if let (Some(h), Some(m)) = (n(&v["daily"][0]), n(&v["daily"][1])) {
            if h < 24 && m < 60 {
                return Ok(Trigger::Daily { hour: h, minute: m });
            }
        }
        if let (Some(d), Some(h), Some(m)) = (n(&v["weekly"][0]), n(&v["weekly"][1]), n(&v["weekly"][2])) {
            if d < 7 && h < 24 && m < 60 {
                return Ok(Trigger::Weekly { day: d, hour: h, minute: m });
            }
        }
        Err("a trigger is {\"every\": minutes}, {\"daily\": [hour, minute]} or {\"weekly\": [day, hour, minute]}".into())
    }
}

/// The latest moment at or before `now_ms` that is `hour:minute` local time (on `day`, if given).
fn latest_slot(now_ms: u64, offset_min: i32, day: Option<u8>, hour: u8, minute: u8) -> u64 {
    let offset = i64::from(offset_min) * MINUTE as i64;
    let local_now = now_ms as i64 + offset;
    let day_start = local_now.div_euclid(DAY as i64) * DAY as i64;
    let mut slot = day_start + (i64::from(hour) * 60 + i64::from(minute)) * MINUTE as i64;
    if slot > local_now {
        slot -= DAY as i64;
    }
    if let Some(want) = day {
        // 1970-01-01 was a Thursday, so day number 0 is weekday 3 with Monday as 0.
        let weekday = |t: i64| (t.div_euclid(DAY as i64) + 3).rem_euclid(7) as u8;
        while weekday(slot) != want {
            slot -= DAY as i64;
        }
    }
    (slot - offset).max(0) as u64
}
