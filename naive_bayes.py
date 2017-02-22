import methods as mt

pos, neg, posProb, negProb,count_value, total_pos, total_neg = mt.train_on_year(2016)

mt.predict_for_2017(pos, neg, posProb, negProb,count_value, total_pos, total_neg)