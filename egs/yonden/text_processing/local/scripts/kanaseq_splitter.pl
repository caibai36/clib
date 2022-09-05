#!/usr/bin/env perl
# Implemented by bin-wu at 21:56 on 2022.06.29

# The configuration file of kana2phone comes from the Kaldi's CSJ script of vocab2dict.pl

use warnings;

use utf8;
use open IO => ":utf8";
use open ":std";

use Getopt::Std;
getopts('khp:'); # The n and h options has no parameters. The p option (with :) has parameter.

$convert_punctuations = 1;

if ((@ARGV != 1 && @ARGV != 0) || $opt_h) {
    die "Usage: kanaseq_splitter.pl [-h (help)] [-k (Kaldi format with uttid in the first column)] [-p PHONELIST (default conf/kana2phone)] kaldi_script_file/stdin\n\nSplit kana sequences in a text or a Kaldi script file.\nKaldi script file should have line format as 'uttid kana_utterance' (e.g., ID227 サギョー イン).\nThe 'kana2phone' file comes from kaldi/egs/csj/s5/local/csj_make_trans/kana2phone\nAdding option '-k' means that the text is Kaldi format with the first column of utterance id (uttid)\n\nExample:\nサガラ 、 リョーカイ =>\nサ ガ ラ <space> <comma> <space> リョー カ イ\n\nNote: ' ' between words converts to <space>, '、' converts to <comma>, '。' converts to <period>, and '？' converts to <question_mark>\nWe print conversion errors into stderr\n";
}

if (!$opt_p) {
    $opt_p = "conf/kana2phone";
}

open(PHONES, $opt_p);
while(<PHONES>) {
    chomp;
    ($kana, $_) = split('\+', $_);
    # $kana2kana{$kana} = $phone;
    $kana2kana{$kana} = $kana # a set of kanas that maps each kana to itself
}

if($convert_punctuations) {
    $kana2kana{"、"} = "<comma>";
    $kana2kana{"。"} = "<period>";
    $kana2kana{"？"} = "<question_mark>";
}

sub kanaseq_splitter($) {
    my($kanaseq) = @_;
    my($flg, $splitted_kanaseq, $syllable, @chars, @syllables);

    $flg = 0;
    $splitted_kanaseq = "";
    $syllable = "";
    @chars = ();
    @syllables = ();

    @chars = split(//, $kanaseq);

    foreach $char (@chars) {
	if ($char =~ /[ァィゥェォャュョ]/) {
	    $syllable .= $char;
	} else {
	    if (!$syllable) {
		$syllable = $char;
	    } else {
		$syllable .= " " . $char;
	    }
	}
    }

    @syllables = split(' ', $syllable);

    foreach $syllable (@syllables) {
	if ($kana2kana{$syllable}) {
	    if ($kana2kana{$syllable} eq "ー") {
		chop($splitted_kanaseq);
	    }
	    $splitted_kanaseq .= $kana2kana{$syllable} . " ";
	} else {
	    $flg = 1;
	    $splitted_kanaseq .= $syllable . " ";
	}
    }

    if ($splitted_kanaseq =~ /^[ー] $/ || $splitted_kanaseq =~ /^[ッ] $/) {
	$flg = 1;
    }

    return ($splitted_kanaseq, $flg);
}

# main function
$line_num = 0;
@error_lines = ();
while (<>) {
    next if /^#/;
    chomp;
    $line_num++;

    if ($opt_k) {
	($uttid, $content) = split('\s', $_, 2);
    } else {
	$content = $_;
    }

    @tokens = split('\s+', $content);

    @kanatokens = ();
    foreach my $token (@tokens) {
    	($splitted_kanatoken, $flag) = &kanaseq_splitter($token);

	if ($flag) {
	    push(@error_lines, "Warning: Line_num: $line_num Token: $token => $splitted_kanatoken in Line: $_");
	}

	$splitted_kanatoken =~ s/\s+$//; # trim white space at the end of the string
    	push(@kanatokens, $splitted_kanatoken);
	
    }

    if ($opt_k) {
	print "$uttid", " ", join(" <space> ", @kanatokens), "\n";
    } else {
	print join(" <space> ", @kanatokens), "\n";
    }
}

print STDERR join("\n", @error_lines);
if (@error_lines != 0) {print STDERR "\n";} # print endline if there is any convertion error

# See: https://stackoverflow.com/questions/17543925/getting-used-only-once-possible-typo-warning-when-aliasing-subroutines
*opt_h if 0; # prevent warning of 'Name "main::opt_h" used only once: possible typo at k2p.pl line 17.'
