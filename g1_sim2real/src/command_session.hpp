#pragma once

#include <cstdint>
#include <string>

// Access under the bridge command-write mutex, including watchdog invalidation.
class CommandSession {
 public:
  enum class Admission { Reject, Begin, Command };
  explicit CommandSession(uint64_t epoch) : epoch_(epoch) {}

  Admission accept(const std::string & id, uint64_t epoch, uint64_t seq,
                   bool begin, bool safe_begin, bool latched, bool active) {
    if (id.empty()) {
      // Preserve the original single-robot protocol until a session client owns it.
      if (begin || epoch != 0 || !id_.empty() || latched || (have_seq_ && seq <= last_seq_)) return Admission::Reject;
    } else {
      if (id.size() != 32 || id.find_first_not_of("0123456789abcdef") != std::string::npos)
        return Admission::Reject;
      if (begin) {
        if (!safe_begin || epoch != epoch_ || (active && !latched) || id == id_)
          return Admission::Reject;
        id_ = id;
        ++epoch_; // A delayed begin packet cannot reset this session.
        last_seq_ = seq;
        have_seq_ = true;
        return Admission::Begin;
      }
      if (id != id_ || epoch != epoch_ || latched || (have_seq_ && seq <= last_seq_))
        return Admission::Reject;
    }
    last_seq_ = seq;
    have_seq_ = true;
    return Admission::Command;
  }

  void invalidate() { ++epoch_; }

  std::string status_json(bool latched) const {
    return "{\"protocol\":1,\"session_id\":\"" + id_ +
        "\",\"epoch\":" + std::to_string(epoch_) +
        ",\"watchdog_latched\":" + (latched ? "true" : "false") + "}";
  }

 private:
  uint64_t epoch_;
  std::string id_;
  uint64_t last_seq_ = 0;
  bool have_seq_ = false;
};
