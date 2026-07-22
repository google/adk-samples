-- Initial Navallist Lite Schema (Flattened)

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1. Functions
CREATE OR REPLACE FUNCTION shortkey_generate() RETURNS text
    LANGUAGE plpgsql
    AS $$
BEGIN
    RETURN substr(translate(encode(gen_random_bytes(9), 'base64'), '+/', '-_'), 1, 11);
END;
$$;

CREATE OR REPLACE FUNCTION notify_changes() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
  payload JSON;
  payload_text TEXT;
  v_trip_id TEXT;
BEGIN
  IF TG_TABLE_NAME = 'trip' THEN
    v_trip_id = NEW.id;
  ELSE
    v_trip_id = NEW.trip_id;
  END IF;

  payload = json_build_object(
    'table', TG_TABLE_NAME,
    'action', TG_OP,
    'data', row_to_json(NEW),
    'trip_id', v_trip_id
  );
  
  payload_text = payload::text;

  IF LENGTH(payload_text) > 7500 THEN
    payload = json_build_object(
      'table', TG_TABLE_NAME,
      'action', TG_OP,
      'trip_id', v_trip_id,
      'truncated', true
    );
    payload_text = payload::text;
  END IF;

  PERFORM pg_notify('db_events', payload_text);
  RETURN NEW;
END;
$$;

-- 2. Tables
CREATE TABLE users (
    id text DEFAULT shortkey_generate() PRIMARY KEY,
    email text NOT NULL UNIQUE,
    google_sub text NOT NULL UNIQUE,
    name text,
    picture text,
    created_at timestamp with time zone DEFAULT now()
);

CREATE TABLE trip (
    id text DEFAULT shortkey_generate() PRIMARY KEY,
    adk_session_id text NOT NULL,
    boat_name text,
    captain_name text,
    departure_time timestamp with time zone,
    status text DEFAULT 'Draft',
    created_at timestamp with time zone DEFAULT now(),
    user_id text,
    trip_type text DEFAULT 'Departing'
);

CREATE TABLE checklist_item (
    id text DEFAULT shortkey_generate() PRIMARY KEY,
    trip_id text NOT NULL REFERENCES trip(id) ON DELETE CASCADE,
    category text NOT NULL,
    name text NOT NULL,
    item_type text,
    is_checked boolean DEFAULT false,
    count_value integer DEFAULT 0,
    location_text text,
    flagged_issue text,
    updated_at timestamp with time zone DEFAULT now(),
    completed_by_user_id text,
    completed_by_name text,
    assigned_to_user_id text,
    assigned_to_name text,
    CONSTRAINT unique_trip_item_name UNIQUE (trip_id, name)
);

CREATE TABLE artifact (
    id text DEFAULT shortkey_generate() PRIMARY KEY,
    trip_id text REFERENCES trip(id) ON DELETE CASCADE,
    checklist_item_id text REFERENCES checklist_item(id) ON DELETE SET NULL,
    filename text NOT NULL,
    mime_type text,
    storage_path text NOT NULL,
    created_at timestamp with time zone DEFAULT now()
);

CREATE TABLE trip_crew (
    id SERIAL PRIMARY KEY,
    trip_id text REFERENCES trip(id) ON DELETE CASCADE,
    user_id text NOT NULL,
    display_name text,
    joined_at timestamp with time zone DEFAULT now(),
    UNIQUE (trip_id, user_id)
);

-- 3. Indexes
CREATE INDEX idx_trip_adk_session_id ON trip(adk_session_id);
CREATE INDEX idx_trip_user_id ON trip(user_id);
CREATE INDEX idx_trip_user_created_at ON trip(user_id, created_at DESC);
CREATE INDEX idx_checklist_trip_id ON checklist_item(trip_id);
CREATE INDEX idx_checklist_trip_cat_name ON checklist_item(trip_id, category, name);
CREATE INDEX idx_checklist_completed_by_user_id ON checklist_item(completed_by_user_id);
CREATE INDEX idx_artifact_trip_id ON artifact(trip_id);
CREATE INDEX idx_artifact_checklist_item_id ON artifact(checklist_item_id);
CREATE INDEX idx_trip_crew_trip_id ON trip_crew(trip_id);

-- 4. Triggers
CREATE TRIGGER trigger_trip_update AFTER INSERT OR DELETE OR UPDATE ON trip FOR EACH ROW EXECUTE FUNCTION notify_changes();
CREATE TRIGGER trigger_checklist_update AFTER INSERT OR DELETE OR UPDATE ON checklist_item FOR EACH ROW EXECUTE FUNCTION notify_changes();