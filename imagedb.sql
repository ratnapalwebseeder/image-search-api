
-- creating images table to store sample images and cad files
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE,
    vector BYTEA,
    image_data BYTEA,
    cad_file BYTEA,
    cad_name VARCHAR
);