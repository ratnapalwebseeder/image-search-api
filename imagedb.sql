
-- creating images table to store sample images
CREATE TABLE images (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL UNIQUE,
    vector BYTEA,
    image_data BYTEA
);