CREATE TABLE IF NOT EXISTS persons (
    id BIGSERIAL PRIMARY KEY,
    person_code VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(200) NOT NULL,
    consent_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS face_enrollments (
    id BIGSERIAL PRIMARY KEY,
    person_id BIGINT NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    model_name VARCHAR(100) NOT NULL,
    embedding BYTEA NOT NULL,
    embedding_dimension SMALLINT NOT NULL CHECK (embedding_dimension > 0),
    photo_data BYTEA NOT NULL,
    photo_sha256 CHAR(64) UNIQUE NOT NULL,
    source_filename VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_face_enrollments_person_id ON face_enrollments(person_id);
