export PYTHONPATH=$PWD
python clib/tests/test_kaldidataloader.py --json-file=clib/tests/data/test_utts.json --padding_tokenid=1 | tee clib/tests/test_outputs/test_kaldidataloader.log
