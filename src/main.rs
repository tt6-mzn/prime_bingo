use itertools::Itertools;
use num::{BigInt, BigRational, Zero, ToPrimitive};
use std::{
    collections::{BTreeSet, HashMap},
    fmt::Debug,
};

struct PrimeFactors {
    eratosthenes: Vec<usize>,
}

impl PrimeFactors {
    fn new(n: usize) -> Self {
        let mut eratosthenes = vec![0usize; n + 1];
        eratosthenes[1] = 1;
        for i in 2..=n {
            if eratosthenes[i] == 0 {
                eratosthenes[i] = i;
                for j in (i..).step_by(i).take_while(|j| *j <= n) {
                    if eratosthenes[j] == 0 {
                        eratosthenes[j] = i;
                    }
                }
            }
        }
        Self { eratosthenes }
    }

    fn prime_factors(&self, mut n: usize) -> Vec<i32> {
        if n == 1 {
            return vec![1];
        }
        let mut factors = vec![];
        while n > 1 {
            factors.push(self.eratosthenes[n] as i32);
            n /= self.eratosthenes[n];
        }
        factors
    }
}

#[derive(Clone)]
enum Expr {
    Value(BigRational),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
}

impl Expr {
    fn evaluate(self) -> Option<BigRational> {
        match self {
            Expr::Value(v) => Some(v),
            Expr::Add(left, right) => match (left.evaluate(), right.evaluate()) {
                (Some(l), Some(r)) => Some(l + r),
                _ => None,
            },
            Expr::Sub(left, right) => match (left.evaluate(), right.evaluate()) {
                (Some(l), Some(r)) => Some(l - r),
                _ => None,
            },
            Expr::Mul(left, right) => match (left.evaluate(), right.evaluate()) {
                (Some(l), Some(r)) => Some(l * r),
                _ => None,
            },
            Expr::Div(left, right) => match (left.evaluate(), right.evaluate()) {
                (Some(l), Some(r)) => {
                    if r.is_zero() {
                        return None;
                    }
                    Some(l / r)
                }
                _ => None,
            },
            Expr::Pow(left, right) => {
                match (left.evaluate(), right.evaluate()) {
                    (Some(l), Some(r)) => {
                        if r.is_integer() && r.to_i32().is_some() {
                            let exp = r.to_i32().unwrap();
                            if exp >= 10000 {
                                return None;
                            }
                            if exp >= 0 {
                                Some(l.pow(exp.abs()))
                            } else {
                                if l.is_zero() {
                                    return None;
                                }
                                Some(BigRational::from_integer(BigInt::from(1)) / l.pow(-exp))
                            }
                        } else {
                            None
                        }
                    },
                    _ => None,
                }
            }
        }
    }

    fn lazy_evaluate(self) -> Expr {
        match self {
            Expr::Value(v) => Expr::Value(v),
            Expr::Add(left, right) => match (left.lazy_evaluate(), right.lazy_evaluate()) {
                (Expr::Value(l), Expr::Value(r)) => Expr::Value(l + r),
                (l, r) => Expr::Add(Box::new(l), Box::new(r)),
            },
            Expr::Sub(left, right) => match (left.lazy_evaluate(), right.lazy_evaluate()) {
                (Expr::Value(l), Expr::Value(r)) => Expr::Value(l - r),
                (l, r) => Expr::Sub(l, r),
            }
            Expr::Mul(, )
        }
    }
}

impl Debug for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Value(v) => write!(f, "{}", v),
            Expr::Add(left, right) => match (left.as_ref(), right.as_ref()) {
                (Expr::Value(_), Expr::Value(_)) => write!(f, "{:?} + {:?}", left, right),
                (Expr::Value(_), _) => write!(f, "{:?} + ({:?})", left, right),
                (_, Expr::Value(_)) => write!(f, "({:?}) + {:?}", left, right),
                (_, _) => write!(f, "({:?}) + ({:?})", left, right),
            },
            Expr::Sub(left, right) => match (left.as_ref(), right.as_ref()) {
                (Expr::Value(_), Expr::Value(_)) => write!(f, "{:?} - {:?}", left, right),
                (Expr::Value(_), _) => write!(f, "{:?} - ({:?})", left, right),
                (_, Expr::Value(_)) => write!(f, "({:?}) - {:?}", left, right),
                (_, _) => write!(f, "({:?}) - ({:?})", left, right),
            },
            Expr::Mul(left, right) => match (left.as_ref(), right.as_ref()) {
                (Expr::Value(_), Expr::Value(_)) => write!(f, "{:?} * {:?}", left, right),
                (Expr::Value(_), _) => write!(f, "{:?} * ({:?})", left, right),
                (_, Expr::Value(_)) => write!(f, "({:?}) * {:?}", left, right),
                (_, _) => write!(f, "({:?}) * ({:?})", left, right),
            },
            Expr::Div(left, right) => match (left.as_ref(), right.as_ref()) {
                (Expr::Value(_), Expr::Value(_)) => write!(f, "{:?} / {:?}", left, right),
                (Expr::Value(_), _) => write!(f, "{:?} / ({:?})", left, right),
                (_, Expr::Value(_)) => write!(f, "({:?}) / {:?}", left, right),
                (_, _) => write!(f, "({:?}) / ({:?})", left, right),
            },
            Expr::Pow(left, right) => write!(f, "pow({:?}, {:?})", left, right),
        }
    }
}

struct PrimeBingo {
    pf: PrimeFactors,
}

impl PrimeBingo {
    // 読み上げられる数が1..=nであるゲームを作成する
    fn new(n: usize) -> Self {
        Self {
            pf: PrimeFactors::new(n),
        }
    }

    fn _enum_exprs(factors: &Vec<i32>) -> Vec<Expr> {
        if factors.len() == 1 {
            return vec![Expr::Value(BigRational::from_integer(BigInt::from(
                factors[0],
            )))];
        }
        let mut exprs = vec![];
        for i in 1..factors.len() {
            let left = Self::_enum_exprs(&factors[..i].to_vec());
            let right = Self::_enum_exprs(&factors[i..].to_vec());
            for (l, r) in left.iter().cartesian_product(right.iter()) {
                exprs.push(Expr::Add(Box::new((*l).clone()), Box::new((*r).clone())));
                exprs.push(Expr::Sub(Box::new((*l).clone()), Box::new((*r).clone())));
                exprs.push(Expr::Sub(Box::new((*r).clone()), Box::new((*l).clone())));
                exprs.push(Expr::Mul(Box::new((*l).clone()), Box::new((*r).clone())));
                exprs.push(Expr::Div(Box::new((*l).clone()), Box::new((*r).clone())));
                exprs.push(Expr::Div(Box::new((*r).clone()), Box::new((*l).clone())));
                exprs.push(Expr::Pow(Box::new((*l).clone()), Box::new((*r).clone())));
                exprs.push(Expr::Pow(Box::new((*r).clone()), Box::new((*l).clone())));
            }
        }
        exprs
    }

    // 数nが読み上げられたとき、その数から作れる数を列挙する
    fn bingo_numbers(&self, n: usize) -> Vec<BigRational> {
        let mut numbers = BTreeSet::new();
        let factors = self.pf.prime_factors(n);
        let exprs = Self::_enum_exprs(&factors);
        for expr in exprs {
            if let Some(v) = expr.evaluate() {
                if v.gt(&BigRational::from_integer(BigInt::from(0))) && v.is_integer() {
                    numbers.insert(v);
                }
            }
        }
        numbers.iter().cloned().collect::<Vec<BigRational>>()
    }
}

fn main() {
    let pg = PrimeBingo::new(99);
    let mut bingo_numbers = HashMap::new();
    for i in 1..=99 {
        bingo_numbers.insert(i, pg.bingo_numbers(i));
    }

    for n in 1..=999 {
        let mut s = Vec::new();
        for k in 1..=99 {
            if bingo_numbers[&k].contains(&BigRational::from_integer(BigInt::from(n))) {
                s.push(k);
            }
        }
        println!("{}: {:?}", n, s);
    }
}

#[test]
fn prime_factors() {
    let pf = PrimeFactors::new(100);
    assert_eq!(pf.prime_factors(100), vec![2, 2, 5, 5]);
    assert_eq!(pf.prime_factors(99), vec![3, 3, 11]);
    assert_eq!(pf.prime_factors(98), vec![2, 7, 7]);
    assert_eq!(pf.prime_factors(97), vec![97]);
}

#[test]
fn expr_eval() {
    // 1 * 2 + 3 * 4
    let expr = Expr::Add(
        Box::new(Expr::Mul(
            Box::new(Expr::Value(BigRational::from_integer(BigInt::from(1)))),
            Box::new(Expr::Value(BigRational::from_integer(BigInt::from(2)))),
        )),
        Box::new(Expr::Mul(
            Box::new(Expr::Value(BigRational::from_integer(BigInt::from(3)))),
            Box::new(Expr::Value(BigRational::from_integer(BigInt::from(4)))),
        )),
    );
    assert_eq!(expr.evaluate(), Some(BigRational::from_integer(BigInt::from(1 * 2 + 3 * 4))));
}
