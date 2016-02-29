clear;
sigmasq = 0.5;
sigma = sqrt(sigmasq);
weights = randn(5,3);
trf = randn(32*25,5);
trl = trf*weights;
trl = trl + sigma*randn(size(trl));
csvwrite('file0.feats.csv',trf)
csvwrite('file0.labels.csv',trl)

tef = randn(32*25,5);
tel = tef*weights;
tel = tel + sigma*randn(size(tel));
csvwrite('file1.feats.csv',tef)
csvwrite('file1.labels.csv',tel)


