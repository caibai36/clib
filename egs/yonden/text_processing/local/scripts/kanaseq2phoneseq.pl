#!/usr/bin/env perl
# Implemented by bin-wu at 21:56 on 2022.05.13

# The function of kanaseq2phoneseq and the configuration file of kana2phone come from the Kaldi's CSJ script of vocab2dict.pl

use warnings;

use utf8;
use open IO => ":utf8";
use open ":std";

use Getopt::Std;
getopts('khp:'); # The n and h options has no parameters. The p option (with :) has parameter.

$convert_punctuations = 1;

if ((@ARGV != 1 && @ARGV != 0) || $opt_h) {
    die "Usage: kanaseq2phnseq.pl [-h (help)] [-k (Kaldi format with uttid in the first column)] [-p PHONELIST (default conf/kana2phone)] kaldi_script_file/stdin\n\nConvert kana sequences to phone sequences in a text or a Kaldi script file.\nKaldi script file should have line format as 'uttid kana_utterance' (e.g., ID227 サギョー イン).\nThe 'kana2phone' file comes from kaldi/egs/csj/s5/local/csj_make_trans/kana2phone\nAdding option '-k' means that the text is Kaldi format with the first column of utterance id (uttid)\n\nNote: ' ' between words converts to <space>, '、' converts to <comma> , '。' converts to <period>, and '？' converts to <question_mark>\nWe print conversion errors into stderr\n";
}

if (!$opt_p) {
    $opt_p = "conf/kana2phone";
}

open(PHONES, $opt_p);
while(<PHONES>) {
    chomp;
    ($kana, $phone) = split('\+', $_); # 'ア+a<space>'
    $kana2phone{$kana} = $phone;
}

if($convert_punctuations) {
    $kana2phone{"、"} = "<comma> "; # add a space at the end (e.g., ア with pronun 'a<space>').
    $kana2phone{"。"} = "<period> ";
    $kana2phone{"？"} = "<question_mark> ";
}

sub kanaseq2phoneseq($) {
    my($kanaseq) = @_;
    my($flg, $phoneseq, $syllable, @chars, @syllables);

    $flg = 0;
    $phoneseq = "";
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
	if ($kana2phone{$syllable}) {
	    if ($kana2phone{$syllable} eq ": ") {
		chop($phoneseq);
	    }
	    $phoneseq .= $kana2phone{$syllable};
	} else {
	    $flg = 1;
	    $phoneseq .= $syllable;
	}
    }

    if ($phoneseq =~ /^: $/ || $phoneseq =~ /^q $/) {
	$flg = 1;
    }

    return ($phoneseq, $flg);
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

    @phonetokens = ();
    foreach my $token (@tokens) {
    	($phoneseq, $flag) = &kanaseq2phoneseq($token);

	if ($flag) {
	    push(@error_lines, "Warning: Line_num: $line_num Token: $token => $phoneseq in Line: $_");
	}

	$phoneseq =~ s/\s+$//; # trim white space at the end of the string
    	push(@phonetokens, $phoneseq);
	
    }

    if ($opt_k) {
	print "$uttid", " ", join(" <space> ", @phonetokens), "\n";
    } else {
	print join(" <space> ", @phonetokens), "\n";
    }
}

print STDERR join("\n", @error_lines);
if (@error_lines != 0) {print("\n");} # print endline if there is any convertion error

# See: https://stackoverflow.com/questions/17543925/getting-used-only-once-possible-typo-warning-when-aliasing-subroutines
*opt_h if 0; # prevent warning of 'Name "main::opt_h" used only once: possible typo at k2p.pl line 17.'
