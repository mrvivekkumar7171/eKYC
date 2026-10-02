CREATE DATABASE IF NOT EXISTS ekyc;
USE ekyc;

DROP TABLE IF EXISTS `user`;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS verification_images;

CREATE TABLE verification_images (
    image_id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    document_image LONGBLOB NOT NULL,
    selfie_image LONGBLOB NOT NULL,
    embedding LONGTEXT NULL,
    id_card_x INT UNSIGNED NULL,
    id_card_y INT UNSIGNED NULL,
    id_card_width INT UNSIGNED NULL,
    id_card_height INT UNSIGNED NULL,
    face_x INT UNSIGNED NULL,
    face_y INT UNSIGNED NULL,
    face_width INT UNSIGNED NULL,
    face_height INT UNSIGNED NULL,
    verification_status VARCHAR(64) NOT NULL DEFAULT 'RECEIVED',
    processing_status VARCHAR(64) NOT NULL DEFAULT 'RECEIVED',
    failure_reason TEXT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (image_id),
    INDEX idx_verification_images_status (verification_status),
    INDEX idx_verification_images_created_at (created_at)
) ENGINE=InnoDB;

CREATE TABLE `user` (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    name VARCHAR(255) NULL,
    father_name VARCHAR(255) NULL,
    dob DATE NULL,
    id_number VARCHAR(255) NOT NULL,
    id_type VARCHAR(32) NOT NULL,
    document_image_id BIGINT UNSIGNED NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id),
    UNIQUE KEY uq_user_id_number_type (id_number, id_type),
    INDEX idx_user_document_image (document_image_id),
    CONSTRAINT fk_user_document_image
        FOREIGN KEY (document_image_id) REFERENCES verification_images (image_id)
) ENGINE=InnoDB;
