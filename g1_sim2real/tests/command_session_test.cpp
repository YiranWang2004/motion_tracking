#include "command_session.hpp"
#include <cassert>

int main() {
  using A = CommandSession::Admission;
  const std::string first(32, 'a'), second(32, 'b');
  CommandSession gate(100);
  // Startup cannot arm with an active command or stale challenge.
  assert(gate.accept(first, 100, 0, true, false, false, false) == A::Reject);
  assert(gate.accept(first, 99, 0, true, true, false, false) == A::Reject);
  assert(gate.accept(first, 100, 0, true, true, false, false) == A::Begin);
  assert(gate.accept(first, 100, 0, true, true, false, false) == A::Reject);
  assert(gate.accept(first, 101, 1, false, false, false, false) == A::Command);
  // A second deploy cannot take over a live controller.
  assert(gate.accept(second, 101, 0, true, true, false, true) == A::Reject);
  assert(gate.accept(first, 101, 3, false, false, false, true) == A::Command);
  assert(gate.accept(first, 101, 2, false, false, false, true) == A::Reject);
  assert(gate.accept(first, 101, 3, false, false, false, true) == A::Reject);
  gate.invalidate();
  assert(gate.accept(first, 101, 4, false, false, true, true) == A::Reject);
  assert(gate.accept(first, 102, 5, true, true, true, true) == A::Reject);
  // Restarting deploy resets sequence to zero, but uses a new session identity.
  assert(gate.accept(second, 102, 0, true, true, true, true) == A::Begin);
  assert(gate.accept(first, 101, 500, false, false, false, false) == A::Reject);
  assert(gate.accept(second, 103, 1, false, false, false, false) == A::Command);
  assert(gate.accept("", 0, 999, false, false, false, true) == A::Reject);
  CommandSession legacy(1);
  assert(legacy.accept("", 0, 10, false, false, false, false) == A::Command);
  assert(legacy.accept("", 0, 9, false, false, false, true) == A::Reject);
  assert(legacy.accept("", 0, 10, false, false, false, true) == A::Reject);
  assert(legacy.accept("", 0, 11, false, false, true, true) == A::Reject);
  assert(legacy.accept(first, 1, 0, true, true, true, true) == A::Begin);
}
