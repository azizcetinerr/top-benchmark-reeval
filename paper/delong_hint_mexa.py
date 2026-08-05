import csv,glob,os,math,numpy as np
def load(p,split='test'):
    d={}
    for r in csv.DictReader(open(p)):
        if r['split']==split: d[r['nctid']]=(float(r['score']),int(r['label']))
    return d
def midrank(x):
    J=np.argsort(x); Z=x[J]; N=len(x); T=np.zeros(N,float); i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N,float); T2[J]=T; return T2
def delong(y,s1,s2):
    order=np.r_[np.where(y==1)[0],np.where(y==0)[0]]     # positives first
    m=int((y==1).sum()); n=len(y)-m
    P=np.vstack([s1,s2])[:,order]
    tx=np.array([midrank(P[r,:m]) for r in (0,1)])
    ty=np.array([midrank(P[r,m:]) for r in (0,1)])
    tz=np.array([midrank(P[r,:])  for r in (0,1)])
    aucs=(tz[:,:m].sum(1)-m*(m+1)/2)/(m*n)
    v01=(tz[:,:m]-tx)/n; v10=1.0-(tz[:,m:]-ty)/m
    S=np.cov(v01)/m+np.cov(v10)/n
    L=np.array([1.,-1.]); d=aucs[0]-aucs[1]; se=math.sqrt(L@S@L)
    z=d/se
    return aucs[0],aucs[1],d,se,math.erfc(abs(z)/math.sqrt(2)),d-1.96*se,d+1.96*se
H=load('preds/preds_hint_phase_III.csv')
best={}
for f in glob.glob('preds/_run_phIII_fix_s*_rho10.05_rho20.01_id*.csv'):
    b=os.path.basename(f); seed=b.split('_s')[1][:4]; i=int(b.split('_id')[1].split('_')[0])
    if seed not in best or i<best[seed][0]: best[seed]=(i,f)
print("A1  HINT vs retrained MEXA-CTP (fixed arm), phase III, paired DeLong")
print("%-6s %-9s %-9s %-9s %-7s %-22s %s"%("seed","AUC HINT","AUC MEXA","delta","SE","95% CI","p"))
ds=[];ps=[];ms=[]
for seed,(i,f) in sorted(best.items()):
    M=load(f); k=sorted(set(H)&set(M))
    y=np.array([H[x][1] for x in k]); a=np.array([H[x][0] for x in k]); b=np.array([M[x][0] for x in k])
    A1,A2,d,se,p,lo,hi=delong(y,a,b); ds.append(d);ps.append(p);ms.append(A2)
    print("%-6s %-9.4f %-9.4f %+-9.4f %-7.4f [%+.4f, %+.4f]  %.3f"%(seed,A1,A2,d,se,lo,hi,p))
print("n=%d | MEXA mean AUC %.4f (sd %.4f) | mean delta %+.4f | p in [%.3f, %.3f]"%(len(k),np.mean(ms),np.std(ms,ddof=1),np.mean(ds),min(ps),max(ps)))
