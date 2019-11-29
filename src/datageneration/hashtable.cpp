#include <boost/functional/hash.hpp>
#include <array>
#include <iostream>
#include <fstream>
#include <unordered_set>
#include <string>


// 64     +    7    +  1  +    4   +    2     +  3 +     3  + 5 = 89
// squares seperator color castling en passant move halfmove spaces
using Board = std::array<char,90>;

struct BoardHash {
    std::size_t operator()(const Board& board) const
    {
        return boost::hash_value(board);
    }
};

void print(const Board& b, std::ostream& out) {
    for (char c : b) {
        if (c == '\0') break;
        out << c;
    }
}

int main(){
    char seperator = ',';
    std::string fen,move;
    std::unordered_set<Board,BoardHash> table;

    char b [90];
    char mv [6];

    Board board;

    std::ofstream file;
    file.open("out.csv");

    while (true) {
        std::cin.getline(b,90);
        std::cin.getline(mv,6);

        if (std::cin.fail()){
            std::cerr << "Error while reading or EOF!";
            break;
        }

        std::copy(std::begin(b),std::end(b),std::begin(board));

        auto[_,inserted] = table.insert(board);

        if (inserted) {
            print(board, file);
            file << seperator << mv << std::endl;
            file.flush();
        }
    }

    file.close();
}