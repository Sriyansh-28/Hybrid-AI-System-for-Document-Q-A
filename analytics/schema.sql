PRAGMA foreign_keys = OFF;

DROP TABLE IF EXISTS order_reviews;
DROP TABLE IF EXISTS order_payments;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS category_translation;
DROP TABLE IF EXISTS sellers;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id           TEXT PRIMARY KEY,
    customer_unique_id    TEXT,
    customer_zip_prefix   TEXT,
    customer_city         TEXT,
    customer_state        TEXT
);

CREATE TABLE sellers (
    seller_id             TEXT PRIMARY KEY,
    seller_zip_prefix     TEXT,
    seller_city           TEXT,
    seller_state          TEXT
);

CREATE TABLE category_translation (
    category_name         TEXT PRIMARY KEY,
    category_name_english TEXT
);

CREATE TABLE products (
    product_id            TEXT PRIMARY KEY,
    category_name         TEXT,
    product_name_length   INTEGER,
    product_desc_length   INTEGER,
    product_photos_qty    INTEGER,
    product_weight_g      INTEGER,
    product_length_cm     INTEGER,
    product_height_cm     INTEGER,
    product_width_cm      INTEGER,
    FOREIGN KEY (category_name) REFERENCES category_translation (category_name)
);

CREATE TABLE orders (
    order_id                       TEXT PRIMARY KEY,
    customer_id                    TEXT,
    order_status                   TEXT,
    order_purchase_timestamp       TEXT,
    order_approved_at              TEXT,
    order_delivered_carrier_date   TEXT,
    order_delivered_customer_date  TEXT,
    order_estimated_delivery_date  TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE TABLE order_items (
    order_id            TEXT,
    order_item_id       INTEGER,
    product_id          TEXT,
    seller_id           TEXT,
    shipping_limit_date TEXT,
    price               REAL,
    freight_value       REAL,
    PRIMARY KEY (order_id, order_item_id),
    FOREIGN KEY (order_id) REFERENCES orders (order_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id),
    FOREIGN KEY (seller_id) REFERENCES sellers (seller_id)
);

CREATE TABLE order_payments (
    order_id             TEXT,
    payment_sequential   INTEGER,
    payment_type         TEXT,
    payment_installments INTEGER,
    payment_value        REAL,
    PRIMARY KEY (order_id, payment_sequential),
    FOREIGN KEY (order_id) REFERENCES orders (order_id)
);

CREATE TABLE order_reviews (
    review_id            TEXT,
    order_id             TEXT,
    review_score         INTEGER,
    review_comment_title TEXT,
    review_comment_message TEXT,
    review_creation_date TEXT,
    review_answer_timestamp TEXT,
    FOREIGN KEY (order_id) REFERENCES orders (order_id)
);

CREATE INDEX idx_orders_customer      ON orders (customer_id);
CREATE INDEX idx_orders_purchase_ts   ON orders (order_purchase_timestamp);
CREATE INDEX idx_orders_status        ON orders (order_status);
CREATE INDEX idx_items_product        ON order_items (product_id);
CREATE INDEX idx_items_seller         ON order_items (seller_id);
CREATE INDEX idx_payments_order       ON order_payments (order_id);
CREATE INDEX idx_reviews_order        ON order_reviews (order_id);
CREATE INDEX idx_products_category    ON products (category_name);
