import logging
import json

# Configure console-only JSON logging with emojis
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)

# Custom JSON log formatter for structured output
class JSONFormatter(logging.Formatter):
    def format(self, record):
        message = record.getMessage()
        try:
            message_dict = json.loads(message)
            if isinstance(message_dict, dict) and "message" in message_dict:
                message = message_dict["message"].encode().decode("unicode_escape")
        except json.JSONDecodeError:
            pass
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "message": message
        }
        return json.dumps(log_entry, ensure_ascii=False)

# Apply JSON formatter to console handler
for handler in logging.getLogger().handlers:
    handler.setFormatter(JSONFormatter())