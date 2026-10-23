"""Logging helper dùng chung cho toàn bộ project.

Tạo 1 logger có format thống nhất, ghi cả ra console và ra file trong
reports/logs/ để tiện xem lại quá trình chạy (đặc biệt là training).
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Trả về logger đã cấu hình sẵn.

    Args:
        name: tên logger, thường dùng __name__ của module gọi.
        log_file: đường dẫn file log tùy chọn. Nếu None, chỉ log ra console.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Đã cấu hình rồi (tránh add handler trùng khi gọi lại nhiều lần)
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
