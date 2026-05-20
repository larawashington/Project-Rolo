use std::fs;

use chrono::Local;
use desktop_pet_lib::vault::events::{
    self, CheckinMethod, DismissContext, EatOutcome, ExperienceEvent, MoodSignal,
};
use desktop_pet_lib::vault::logger::ExperienceLogger;
use tempfile::TempDir;

#[test]
fn round_trip_every_variant_through_the_logger() {
    let tmp = TempDir::new().unwrap();
    let events_dir = tmp.path().join("events");
    fs::create_dir_all(&events_dir).unwrap();

    let logger = ExperienceLogger::new(events_dir.clone());

    let ts = Local::now();
    let events = vec![
        ExperienceEvent::Chat {
            event_id: events::new_id(),
            ts,
            session: "session-1".into(),
            user: "hi".into(),
            rolo: "hi back".into(),
            mood_signal: None,
        },
        ExperienceEvent::Chat {
            event_id: events::new_id(),
            ts,
            session: "session-1".into(),
            user: "I'm tired".into(),
            rolo: "*sniff sniff*".into(),
            mood_signal: Some(MoodSignal::Negative),
        },
        ExperienceEvent::Dismiss {
            event_id: events::new_id(),
            ts,
            context: DismissContext::IdleSpeech,
            times_dismissed_session: 2,
        },
        ExperienceEvent::Eat {
            event_id: events::new_id(),
            ts,
            files: vec!["old_migration.sql".into()],
            outcome: EatOutcome::Satisfied,
            bytes: 4200,
        },
        ExperienceEvent::Checkin {
            event_id: events::new_id(),
            ts,
            question: "how's your afternoon?".into(),
            response: "good".into(),
            method: CheckinMethod::Button,
        },
        ExperienceEvent::Drag {
            event_id: events::new_id(),
            ts,
            duration_ms: 1500,
        },
        ExperienceEvent::IdleSpeech {
            event_id: events::new_id(),
            ts,
            text: "nice focus.".into(),
            dismissed: false,
        },
        ExperienceEvent::Report {
            event_id: events::new_id(),
            ts,
            message_id: 4129,
            rolo_text: "...the offending text...".into(),
        },
        ExperienceEvent::SessionEnd {
            event_id: events::new_id(),
            ts,
            idle_total_ms: 1_200_000,
            interactions: 12,
        },
    ];

    for e in &events {
        logger.log(e);
    }
    logger.flush();

    let today = Local::now().date_naive();
    let path = events_dir.join(format!("{today}.jsonl"));
    let contents = fs::read_to_string(&path).expect("event log file should exist");
    let lines: Vec<&str> = contents.lines().collect();

    assert_eq!(
        lines.len(),
        events.len(),
        "one line per event, got {} lines for {} events",
        lines.len(),
        events.len()
    );

    for (i, line) in lines.iter().enumerate() {
        let _: ExperienceEvent = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("line {i} failed to parse: {e}\n  line: {line}"));
    }
}
