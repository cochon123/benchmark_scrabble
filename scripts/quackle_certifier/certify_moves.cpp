#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "alphabetparameters.h"
#include "board.h"
#include "datamanager.h"
#include "game.h"
#include "generator.h"
#include "lexiconparameters.h"
#include "move.h"
#include "player.h"
#include "playerlist.h"
#include "rack.h"
#include "trademarkedboards.h"

static std::vector<std::string> split(const std::string &value, char delimiter) {
  std::vector<std::string> result;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, delimiter)) result.push_back(item);
  return result;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: certify_moves DICTIONARY_PREFIX MAX_MOVES\n";
    return 2;
  }
  const int max_moves = std::stoi(argv[2]);
  Quackle::DataManager data;
  data.setBoardParameters(new ScrabbleBoard());
  data.lexiconParameters()->loadDawg(std::string(argv[1]) + ".dawg");
  data.lexiconParameters()->loadGaddag(std::string(argv[1]) + ".gaddag");
  if (!data.lexiconParameters()->hasSomething()) {
    std::cerr << "failed to load dictionary\n";
    return 2;
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    const auto fields = split(line, '\t');
    if (fields.size() != 3) {
      std::cerr << "bad input row\n";
      return 2;
    }
    Quackle::PlayerList players;
    players.push_back(Quackle::Player("audit", Quackle::Player::HumanPlayerType, 0));
    Quackle::GamePosition position(players);
    position.incrementTurn(nullptr);

    Quackle::Board board;
    board.prepareEmptyBoard();
    if (!fields[2].empty() && fields[2] != "-") {
      for (const auto &cell : split(fields[2], ';')) {
        const auto parts = split(cell, ',');
        if (parts.size() != 4) return 2;
        const int row = std::stoi(parts[0]);
        const int col = std::stoi(parts[1]);
        std::string letter = parts[2];
        auto encoded = data.alphabetParameters()->encode(letter);
        if (parts[3] == "1") encoded = Quackle::String::setBlankness(encoded);
        board.makeMove(Quackle::Move::createPlaceMove(row, col, true, encoded));
      }
    }
    position.setBoard(board);
    position.setCurrentPlayerRack(
        Quackle::Rack(data.alphabetParameters()->encode(fields[1])), false);
    position.ensureBoardIsPreparedForAnalysis();

    Quackle::Generator generator(position);
    generator.kibitz(max_moves, Quackle::Generator::CannotExchange);
    int best_score = -1;
    int best_count = 0;
    for (const auto &move : generator.kibitzList()) {
      if (move.action != Quackle::Move::Place) continue;
      if (move.score > best_score) {
        best_score = move.score;
        best_count = 1;
      } else if (move.score == best_score) {
        ++best_count;
      }
    }
    std::cout << fields[0] << '\t' << best_score << '\t' << best_count << '\n';
  }
  return 0;
}
