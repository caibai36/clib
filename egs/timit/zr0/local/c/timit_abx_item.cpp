#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <algorithm>
using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cerr << "Create the abx_item from the normalized anotation of timit format." << endl;
    cerr << "And output the item format for the file for abx test." << endl << endl;
    cerr << "Usage: ./local/timit_abx_item <normalized-timit-format-file> " << endl;
    cerr << "eg: ./local/timit_abx_item data/test_phn_timit_normal/MCMB0_SI638.PHN" << endl;
    cerr << "NOTE: normalize-timit-format: <start-time> <end_time> <label>" << endl;
    cerr << "      item-format: #file onset offset #phone context talker" << endl;
    cerr << "               eg: s2001b 270.062 270.992 aa SIL-SIL 20" << endl;
    return 1;
  }
  
  ifstream in(argv[1]);
  if (!in) {
    cerr << "Fail to open the file: " << argv[1] << endl;
    return 1;
  }

  /* From file to get the utterance and the speaker. */
  int pos;
  string file(argv[1]);
  string reversed(file.rbegin(), file.rend());
  if ((pos = reversed.find('/')) != string::npos)
    file = file.substr(file.size() - pos);
     
  int p = file.find(".PHN");
  int h = file.find("_");
  assert(h != string::npos && p != string::npos);
  
  string spk = file.substr(0, h);
  string utt = file.substr(0, p);

  /* Print out the item format. */
  string label;
  double start, end;
  vector<string> labels;
  vector<double> starts, ends;
  while(in >> start >> end >> label) {
    labels.push_back(label);
    starts.push_back(start);
    ends.push_back(end);
  }

  assert(labels.size() == starts.size() && labels.size() == ends.size());
  assert(labels.size() >= 3);

  for (int i = 1; i < labels.size() - 1; ++i) 
    cout << utt << " " << starts[i - 1] << " " << ends[i + 1] << " " << labels[i] << " " << labels[i - 1] << "-" << labels[i + 1] << " " << spk << endl;
}
 
