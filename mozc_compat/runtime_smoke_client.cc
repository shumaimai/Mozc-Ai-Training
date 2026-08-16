// Local runtime smoke test for the installed Mozc IPC server.
// It sends only fixed synthetic input and never prints candidate text.

#include <algorithm>
#include <iostream>
#include <memory>
#include <string_view>

#include "base/init_mozc.h"
#include "client/client.h"
#include "client/client_interface.h"
#include "protocol/commands.pb.h"

int main(int argc, char** argv) {
  mozc::InitMozc(argv[0], &argc, &argv);
  std::unique_ptr<mozc::client::ClientInterface> client =
      mozc::client::ClientFactory::NewClient();
  client->set_suppress_error_dialog(true);
  if (!client->EnsureConnection()) {
    std::cerr << "RUNTIME_SMOKE connection_failed\n";
    return 2;
  }

  mozc::commands::SessionCommand turn_on;
  turn_on.set_type(mozc::commands::SessionCommand::TURN_ON_IME);
  turn_on.set_composition_mode(mozc::commands::HIRAGANA);
  mozc::commands::Output mode_output;
  if (!client->SendCommand(turn_on, &mode_output)) {
    std::cerr << "RUNTIME_SMOKE turn_on_failed\n";
    return 3;
  }

  mozc::commands::Context context;
  context.set_preceding_text("駅に到着して");
  mozc::commands::Output output;
  constexpr std::string_view kSyntheticKeys = "kisya";
  for (const char key_code : kSyntheticKeys) {
    mozc::commands::KeyEvent key;
    key.set_key_code(static_cast<unsigned char>(key_code));
    key.set_mode(mozc::commands::HIRAGANA);
    if (!client->SendKeyWithContext(key, context, &output)) {
      std::cerr << "RUNTIME_SMOKE send_key_failed\n";
      return 4;
    }
  }

  mozc::commands::KeyEvent space;
  space.set_special_key(mozc::commands::KeyEvent::SPACE);
  space.set_mode(mozc::commands::HIRAGANA);
  if (!client->SendKeyWithContext(space, context, &output)) {
    std::cerr << "RUNTIME_SMOKE conversion_failed\n";
    return 5;
  }

  const int candidate_count =
      std::max(output.all_candidate_words().candidates_size(),
               output.candidate_window().candidate_size());
  mozc::commands::SessionCommand reset;
  reset.set_type(mozc::commands::SessionCommand::RESET_CONTEXT);
  mozc::commands::Output reset_output;
  client->SendCommand(reset, &reset_output);
  std::cout << "RUNTIME_SMOKE ok candidate_count=" << candidate_count
            << " consumed=" << output.consumed()
            << " preedit_segments=" << output.preedit().segment_size()
            << " has_candidate_window=" << output.has_candidate_window()
            << " candidate_category="
            << static_cast<int>(output.candidate_window().category())
            << " has_result=" << output.has_result() << "\n";
  return candidate_count > 0 ? 0 : 6;
}
