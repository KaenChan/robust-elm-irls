#!
# author: Jun Xu and Tie-Yan Liu#!
use strict;

#hash table for NDCG,
my %hsNdcgRelScore = (  "2", 3,
                        "1", 1,
                        "0", 0,
                    );

#hash table for Precision@N and MAP
my %hsPrecisionRel = ("2", 1,
                      "1", 1,
                      "0", 0
                );

my $iMaxPosition = 10;

my $argc = $#ARGV+1;
if($argc != 4)
{
		print "Invalid command line.\n";
		print "Usage: perl Eval.pl argv[1] argv[2] argv[3] argv[4]\n";
		print "argv[1]: feature file \n";
		print "argv[2]: prediction file\n";
		print "argv[3]: result (output) file\n";
		print "argv[4]: flag. If flag equals 1, output the evaluation results per query; if flag equals 0, simply output the average results.\n";
		exit -1;
}
my $fnFeature = $ARGV[0];
my $fnPrediction = $ARGV[1];
my $fnResult = $ARGV[2];
my $flag = $ARGV[3];
if($flag != 1 && $flag != 0)
{
	print "Invalid command line.\n";
	print "Usage: perl Eval.pl argv[1] argv[2] argv[3] argv[4]\n";
	print "Flag should be 0 or 1\n";
	exit -1;
}

my %hsQueryDocLabelScore = ReadInputFiles($fnFeature, $fnPrediction);
my %hsQueryEval = EvalQuery(\%hsQueryDocLabelScore);
OuputResults($fnResult, %hsQueryEval);


sub OuputResults
{
    my ($fnOut, %hsResult) = @_;
    open(FOUT, ">$fnOut");
    my @Ndcg;
    my @PAtN;
    my $MAP;

    my @qids = sort{$a <=> $b} keys(%hsResult);
    my $numQuery = @qids;
    #Precision@N
    my @prec;
    for(my $i = 0; $i < $#qids + 1; $i ++)
    {
        my $qid = $qids[$i];
        my @pN = @{$hsResult{$qid}{"PatN"}};
        if ($flag == 1)
        {
            print FOUT "precision of query$i:\t";
            print FOUT join("\t", @pN);
            print FOUT "\t\n";
        }
        for(my $iPos = 0; $iPos < $#pN + 1; $iPos ++)
        {
            $prec[$iPos] += $pN[$iPos];
        }
    }
    for(my $iPos = 0; $iPos < $#prec + 1; $iPos ++)
    {
        $prec[$iPos] /= ($#qids + 1);
    }
    print FOUT "precision:\t";
    print FOUT join("\t", @prec);
    print FOUT "\t\n\n";
    
    #MAP
    my $Map = 0;
    for(my $i = 0; $i < $#qids + 1; $i ++)
    {
        my $qid = $qids[$i];
        my $avgP = $hsResult{$qid}{"MAP"};
        if ($flag == 1)
        {
            print FOUT "Map of query$i:\t";
            print FOUT $avgP;
            print FOUT "\n";
        }
        $Map += $avgP;
    }
    $Map /= ($#qids + 1);
    print FOUT "MAP:\t";
    print FOUT "$Map";
    print FOUT "\n\n";
    
    #NDCG
    my @ndcg;
    for(my $i = 0; $i < $#qids + 1; $i ++)
    {
        my $qid = $qids[$i];
        my @ndcg_q = @{$hsResult{$qid}{"NDCG"}};
        if ($flag == 1)
        {
            print FOUT "NDCG of query$i:\t";
            print FOUT join("\t", @ndcg_q);
            print FOUT "\t\n";
        }
        for(my $iPos = 0; $iPos < $#ndcg_q + 1; $iPos ++)
        {
            $ndcg[$iPos] += $ndcg_q[$iPos];
        }
    }
    print FOUT "NDCG:\t";
    for(my $iPos = 0; $iPos < $#ndcg + 1; $iPos ++)
    {
        $ndcg[$iPos] /= ($#qids + 1);
        print FOUT "$ndcg[$iPos]\t";
    }
    print FOUT "\n\n";
    close(FOUT);
}

sub EvalQuery
{
    my $pHash = $_[0];
    my %hsResults;
    
    my @qids = sort{$a <=> $b} keys(%$pHash);
    for(my $i = 0; $i < @qids; $i ++)
    {
        my $qid = $qids[$i];
        my @tmpDid = sort{$$pHash{$qid}{$a}{"lineNum"} <=> $$pHash{$qid}{$b}{"lineNum"}} keys(%{$$pHash{$qid}});
        my @docids = sort{$$pHash{$qid}{$b}{"pred"} <=> $$pHash{$qid}{$a}{"pred"}} @tmpDid;
        my @rates;

        for(my $iPos = 0; $iPos < $#docids + 1; $iPos ++)
        {
            $rates[$iPos] = $$pHash{$qid}{$docids[$iPos]}{"label"};
        }

        my $map  = MAP(@rates);
        my @PAtN = PrecisionAtN($iMaxPosition, @rates);
        my @Ndcg = NDCG($iMaxPosition, @rates);
        
        
        @{$hsResults{$qid}{"PatN"}} = @PAtN;
        $hsResults{$qid}{"MAP"} = $map;
        @{$hsResults{$qid}{"NDCG"}} = @Ndcg;
    }
    return %hsResults;
}

sub ReadInputFiles
{
    my ($fnFeature, $fnPred) = @_;
    my %hsQueryDocLabelScore;
    
    if(!open(FIN_Feature, $fnFeature))
	{
		print "Invalid command line.\n";
		print "Open \$fnFeature\" failed.\n";
		exit -2;
	}
	if(!open(FIN_Pred, $fnPred))
	{
		print "Invalid command line.\n";
		print "Open \"$fnPred\" failed.\n";
		exit -2;
	}

    my $lineNum = 0;
    while(defined(my $lnFea = <FIN_Feature>))
    {
        $lineNum ++;
        chomp($lnFea);
        my $predScore = <FIN_Pred>;
        if (!defined($predScore))
        {
            print "Error to read $fnPred at line $lineNum.\n";
            exit -2;
        }
        chomp($predScore);
# modified by Jun Xu, 2008-9-9
# Labels may have more than 3 levels
# qid and docid may not be numeric

       if ($lnFea =~ m/^([0-2]) qid\:(\d+).*?\#docid = (\d+)/)
       # if ($lnFea =~ m/^([0-2]) qid\:(\d+).*?\#docid = (\d+)$/)
        # if ($lnFea =~ m/^(\d+) qid\:([^\s]+).*?\#docid = ([^\s]+)$/)
        {
            my $label = $1;
            my $qid = $2;
            my $did = $3;
            $hsQueryDocLabelScore{$qid}{$did}{"label"} = $label;
            $hsQueryDocLabelScore{$qid}{$did}{"pred"} = $predScore;
            $hsQueryDocLabelScore{$qid}{$did}{"lineNum"} = $lineNum;
        }
        else
        {
            print "Error to parse $fnFeature at line $lineNum:\n$lnFea\n";
            exit -2;
        }
    }
    close(FIN_Feature);
    close(FIN_Pred);
    return %hsQueryDocLabelScore;
}


sub PrecisionAtN
{
    my ($topN, @rates) = @_;
    my @PrecN;
    my $numRelevant = 0;
    for(my $iPos = 0;  $iPos < $topN && $iPos < $#rates + 1; $iPos ++)
    {
        $numRelevant ++ if ($hsPrecisionRel{$rates[$iPos]} == 1);
        $PrecN[$iPos] = $numRelevant / ($iPos + 1);
    }
    return @PrecN;
}

sub MAP
{
    my @rates = @_;

    my $numRelevant = 0;
    my $avgPrecision = 0.0;
    for(my $iPos = 0; $iPos < $#rates + 1; $iPos ++)
    {
        if ($hsPrecisionRel{$rates[$iPos]} == 1)
        {
            $numRelevant ++;
            $avgPrecision += ($numRelevant / ($iPos + 1));
        }
    }
    return 0.0 if ($numRelevant == 0);
    #return sprintf("%.4f", $avgPrecision / $numRelevant);
    return $avgPrecision / $numRelevant;
}

sub DCG
{
    my ($topN, @rates) = @_;
    my @dcg;
    
    $dcg[0] = $hsNdcgRelScore{$rates[0]};
    
    for(my $iPos = 0; $iPos < $topN && $iPos < $#rates + 1; $iPos ++) {
        printf "%d ", $rates[$iPos] ;
    }
    print "\n";

    for(my $iPos = 1; $iPos < $topN && $iPos < $#rates + 1; $iPos ++)
    {
        if ($iPos < 2)
        {
            $dcg[$iPos] = $dcg[$iPos - 1] + $hsNdcgRelScore{$rates[$iPos]};
        }
        else
        {
            $dcg[$iPos] = $dcg[$iPos - 1] + ($hsNdcgRelScore{$rates[$iPos]} * log(2.0) / log($iPos + 1.0));
        }
        printf "%.4f ", log(2.0) / log($iPos + 1.0) ;
    }
    print "\n";
    for(my $iPos = 0; $iPos < $topN && $iPos < $#rates + 1; $iPos ++) {
        printf "%.4f ", $dcg[$iPos] ;
    }
    print "\n";
    return @dcg;
}
sub NDCG
{
    my ($topN, @rates) = @_;
    my @ndcg;
    my @dcg = DCG($topN, @rates);
    my @stRates = sort {$hsNdcgRelScore{$b} <=> $hsNdcgRelScore{$a}} @rates;
    my @bestDcg = DCG($topN, @stRates);
    
    for(my $iPos =0; $iPos < $topN && $iPos < $#rates + 1; $iPos ++)
    {
        $ndcg[$iPos] = 0;
        $ndcg[$iPos] = $dcg[$iPos] / $bestDcg[$iPos] if ($bestDcg[$iPos] != 0);
        #$ndcg[$iPos] = sprintf("%.4f", $ndcg[$iPos]);
    }
    return @ndcg;
}

