CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    model_type VARCHAR(50) PRIMARY KEY,
    mAP FLOAT,
    inference_time FLOAT,
    throughput FLOAT,
    `precision` FLOAT,
    count INT,
    result_image_path VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS detection_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_type VARCHAR(50),
    image_path VARCHAR(255),
    inference_time FLOAT,
    mAP FLOAT,
    `precision` FLOAT,
    throughput FLOAT,
    detected_objects TEXT,
    detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trained_models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_type VARCHAR(50),
    dataset_path VARCHAR(255),
    weights_path VARCHAR(255),
    epochs INT,
    username VARCHAR(50),
    training_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);