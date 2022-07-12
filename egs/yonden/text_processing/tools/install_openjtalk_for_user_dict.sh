if [ $# -eq 0 ]; then
    echo "Usage:"
    echo "$0 <dir_to_install>"
    echo "e.g.:"
    echo "$0 ."
    echo "$0 tools/openjtalk_with_user_dict"
    exit 1
fi

cur_dir=$PWD
install_dir=$1
mkdir -p $install_dir

cd $install_dir
source /project/nakamura-lab08/Work/bin-wu/share/tools/gcc/path.sh # gcc 5.4.0, needed for successfully compiling mecab C code.
git clone https://github.com/Yosshi999/pyopenjtalk.git
cd pyopenjtalk/
git branch -a
git checkout PR-user-dic
git submodule update --recursive --init
pip install -e .
# test existence of function "create_user_dict"
# python -c 'import pyopenjtalk; print(pyopenjtalk.create_user_dict)'
cd $cur_dir
