package com.example.api;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.springframework.http.HttpStatus;


@AllArgsConstructor
@Getter
public enum ErrorCode {
    INVALID_MIME_TYPE(HttpStatus.UNSUPPORTED_MEDIA_TYPE, "invalid mime type"),
    INVALID_UPLOAD_FILE(HttpStatus.BAD_REQUEST, "invalid file")
    ;
    private final HttpStatus httpStatus;
    private final String message;

}
