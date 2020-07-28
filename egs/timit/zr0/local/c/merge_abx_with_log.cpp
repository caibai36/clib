/* 
 * Merge two abx files with logs.
 * if the two abx line have the common time indice,
 * print the corresponding label pair.
 * 
 * Usage: ./merge_abx abx_file1 abx_file2
 */

#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 4) {
    cerr << "Merge two abx files. " << endl
	 << "if the two abx line have the common time indice," << endl
	 << "print the corresponding label pair with logs." << endl << endl
	 << "Usage: ./merge_abx abx_file1 abx_file2 log_dir" << endl
         << "eg: ./local/merge_abx data/test_phn_abx/FAKS0_SI1573.PHN.abx dpgmm/test/data/03.raw.vtln.cmvn.deltas.mfcc/cluster_label/FAKS0_SI1573.clabel.abx dpgmm/test/data/03.raw.vtln.cmvn.deltas.mfcc" << endl << endl;
    return 1;
  }

  ifstream in1(argv[1]), in2(argv[2]);
  if (!in1 && !in2) cerr << "File not found!" << endl;
  ofstream log((string(argv[3]) + "/log.txt").c_str(), log.out | log.app);

  map<double, string> m1, m2;
  vector<double> t, t1, t2;
  double time;
  string label;
  while (in1 >> time >> label) {
    m1.insert(make_pair(time, label));
    t1.push_back(time);
  }

  while (in2 >> time >> label) { 
    m2.insert(make_pair(time, label));
    t2.push_back(time);
  }

  t = t1.size() >= t2.size()? t1 : t2;

  /* All mistmatching frames will be printed out to log. */
  log << "Matching abx frames. " << endl;
  for (vector<double>::iterator itr = t.begin(); itr != t.end(); ++itr) {
    if(m1.find(*itr) != m1.end() && m2.find(*itr) != m2.end()) cout << *itr << " " << m1[*itr] << " " << m2[*itr] << endl;
    if(m1.find(*itr) != m1.end() && m2.find(*itr) == m2.end()) log << argv[1] << " " << *itr << " " << m1[*itr] << endl;
    if(m1.find(*itr) == m1.end() && m2.find(*itr) != m2.end()) log << argv[2] << " " << *itr << " " << m2[*itr] << endl;
  }
}
