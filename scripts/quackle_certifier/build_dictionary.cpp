#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

#include <QString>

#include "dawgfactory.h"
#include "gaddagfactory.h"

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: build_dictionary WORDS ALPHABET OUT_DAWG OUT_GADDAG\n";
    return 2;
  }
  std::ifstream words(argv[1]);
  if (!words) {
    std::cerr << "cannot open word list: " << argv[1] << "\n";
    return 2;
  }

  DawgFactory dawg(QString::fromUtf8(argv[2]));
  GaddagFactory gaddag(argv[2]);
  std::string word;
  while (words >> word) {
    dawg.pushWord(word, true, 1);
    gaddag.pushWord(word);
  }
  dawg.generate();
  gaddag.sortWords();
  gaddag.generate();
  dawg.writeIndex(argv[3]);
  gaddag.writeIndex(argv[4]);
  std::cout << "words=" << dawg.encodableWords()
            << " dawg_nodes=" << dawg.nodeCount()
            << " gaddag_nodes=" << gaddag.nodeCount() << "\n";
  return dawg.unencodableWords() || gaddag.unencodableWords();
}
