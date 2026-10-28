"""Faithful reference of EchoWorld_Demo.html act() world logic (excluding position/UI)."""
import random
ACTIONS=['rain','sun','wind','seed','dam','open']
FIELDS=['rain','sun','wind','wet','eco','seed','grown','dam','gate','lit']
BOOL=['rain','sun','wind','seed','grown','dam','gate','lit']

def fresh():
    return dict(rain=False,sun=False,wind=False,wet=10,eco=10,seed=False,grown=False,dam=False,gate=False,lit=False)

def transition(state, action):
    if action not in ACTIONS: raise ValueError(action)
    s=state.copy()
    if action=='rain':
        s['rain']=True;s['sun']=False;s['wet']=min(100,s['wet']+34)
    elif action=='sun':
        s['sun']=True;s['rain']=False;s['wet']=max(0,s['wet']-9)
    elif action=='wind':s['wind']=not s['wind']
    elif action=='seed':
        if not s['seed']:s['seed']=True
    elif action=='dam':s['dam']=True;s['wet']=min(100,s['wet']+14)
    elif action=='open':
        if s['dam'] and s['grown']:s['gate']=True
    if s['seed'] and s['wet']>=45 and s['sun'] and not s['grown']:
        s['grown']=True;s['eco']+=60
    if s['gate'] and s['grown'] and not s['lit']:
        s['lit']=True;s['eco']=100
    return s

def sample_episodes(n, seed):
    rng=random.Random(seed)
    for episode in range(n):
        s=fresh()
        for t in range(rng.randint(5,24)):
            a=rng.choice(ACTIONS)
            nxt=transition(s,a)
            yield {'episode':episode,'step':t,'state':s.copy(),'action':a,'next_state':nxt.copy()}
            s=nxt

def random_reachable(n, seed):
    rng=random.Random(seed)
    for i in range(n):
        s=fresh()
        for _ in range(rng.randint(1,30)):
            s=transition(s,rng.choice(ACTIONS))
        a=rng.choice(ACTIONS)
        yield {'episode':i,'step':0,'state':s,'action':a,'next_state':transition(s,a)}

def encode(s,a):
    return [float(s[k])/100 if k in ('wet','eco') else float(s[k]) for k in FIELDS]+[float(a==k) for k in ACTIONS]

def target(s):
    return [float(s[k])/100 if k in ('wet','eco') else float(s[k]) for k in FIELDS]

def decode(values):
    return {k: (round(min(100,max(0,values[i]*100)),1) if k in ('wet','eco') else bool(values[i]>=.5)) for i,k in enumerate(FIELDS)}
