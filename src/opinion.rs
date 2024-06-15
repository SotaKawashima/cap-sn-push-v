pub mod gen2;
pub mod paramter;

use crate::info::InfoContent;
use paramter::{ConditionParams, DependentParam, SimplexDist, SimplexParam};

use approx::UlpsEq;
use num_traits::{Float, FromPrimitive, NumAssign, ToPrimitive};
use rand::Rng;
use rand_distr::{uniform::SampleUniform, Distribution, Exp1, Open01, Standard, StandardNormal};
use serde_with::{serde_as, TryFromInto};
use std::{fmt, iter::Sum, ops::AddAssign};
use subjective_logic::{
    domain::{Domain, DomainConv},
    impl_domain,
    iter::FromFn,
    marr_d1,
    mul::{
        labeled::{OpinionD1, OpinionD2, OpinionD3, SimplexD1},
        InverseCondition, MergeJointConditions2,
    },
    multi_array::labeled::{MArrD1, MArrD2, MArrD3},
    ops::{Deduction, Discount, Fuse, FuseAssign, FuseOp, Product2, Product3, Projection},
};
use tracing::{debug, span, Level};

#[derive(Debug)]
pub struct Psi;
impl_domain!(Psi = 2);

#[derive(Debug)]
pub struct Phi;
impl_domain!(Phi = 2);

#[derive(Debug)]
pub struct M;
impl_domain!(M = 2);

#[derive(Debug)]
pub struct O;
impl_domain!(O = 2);

#[derive(Debug)]
pub struct A;
impl_domain!(A = 2);

#[derive(Debug)]
pub struct B;
impl_domain!(B = 2);

#[derive(Debug)]
pub struct H;
impl_domain!(H = 2);

#[derive(Debug)]
pub struct Theta;
impl_domain!(Theta from H);

#[derive(Debug)]
pub struct Thetad;
impl_domain!(Thetad from H);

#[derive(Debug)]
pub struct FO;
impl_domain!(FO from O);

#[derive(Debug)]
pub struct FM;
impl_domain!(FM from M);

#[derive(Debug)]
pub struct FPhi;
impl_domain!(FPhi from Phi);

#[derive(Debug)]
pub struct FPsi;
impl_domain!(FPsi from Psi);

#[derive(Debug)]
pub struct FB;
impl_domain!(FB from B);

#[derive(Debug)]
pub struct FH;
impl_domain!(FH from H);

#[derive(Debug)]
pub struct KPsi;
impl_domain!(KPsi from Psi);

#[derive(Debug)]
pub struct KPhi;
impl_domain!(KPhi from Phi);

#[derive(Debug)]
pub struct KM;
impl_domain!(KM from M);

#[derive(Debug)]
pub struct KO;
impl_domain!(KO from O);

#[derive(Debug)]
pub struct KH;
impl_domain!(KH from H);

#[derive(Debug)]
pub struct KB;
impl_domain!(KB from B);

#[derive(Debug, serde::Deserialize)]
pub struct GlobalBaseRates<V> {
    pub base: BaseRates<V>,
    pub friend: FriendBaseRates<V>,
    pub social: SocialBaseRates<V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct BaseRates<V> {
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub psi: MArrD1<Psi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub phi: MArrD1<Phi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub m: MArrD1<M, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub o: MArrD1<O, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub a: MArrD1<A, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub b: MArrD1<B, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub h: MArrD1<H, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub theta: MArrD1<Theta, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub thetad: MArrD1<Thetad, V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct FriendBaseRates<V> {
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub fpsi: MArrD1<FPsi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub fphi: MArrD1<FPhi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub fm: MArrD1<FM, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub fo: MArrD1<FO, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub fb: MArrD1<FB, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub fh: MArrD1<FH, V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct SocialBaseRates<V> {
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub kpsi: MArrD1<KPsi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub kphi: MArrD1<KPhi, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub km: MArrD1<KM, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub ko: MArrD1<KO, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub kb: MArrD1<KB, V>,
    #[serde_as(as = "TryFromInto<Vec<V>>")]
    pub kh: MArrD1<KH, V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
pub struct InitialOpinions<V: Float + AddAssign + UlpsEq> {
    pub base: InitialBaseSimplexes<V>,
    pub friend: InitialFriendSimplexes<V>,
    pub social: InitialSocialSimplexes<V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialBaseSimplexes<V: Float + AddAssign + UlpsEq> {
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub psi: SimplexD1<Psi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub phi: SimplexD1<Phi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub m: SimplexD1<M, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub o: SimplexD1<O, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub h_if_phi1_psi1: SimplexD1<H, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub h_if_phi1_b1: SimplexD1<H, V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialFriendSimplexes<V: Float + AddAssign + UlpsEq> {
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub fphi: SimplexD1<FPhi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub fm: SimplexD1<FM, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub fo: SimplexD1<FO, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub fh_if_fphi1_fpsi1: SimplexD1<FH, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub fh_if_fphi1_fb1: SimplexD1<FH, V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize, Clone)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialSocialSimplexes<V: Float + AddAssign + UlpsEq> {
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub kphi: SimplexD1<KPhi, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub km: SimplexD1<KM, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub ko: SimplexD1<KO, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub kh_if_kphi1_kpsi1: SimplexD1<KH, V>,
    #[serde_as(as = "TryFromInto<(Vec<V>, V)>")]
    pub kh_if_kphi1_kb1: SimplexD1<KH, V>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
pub struct InitialConditions<V>
where
    V: Float + UlpsEq + AddAssign,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    pub base: InitialBaseConditions<V>,
    pub friend: InitialFriendConditions<V>,
    pub social: InitialSocialConditions<V>,
}

pub struct SampleInitialConditions<V> {
    base: SampleInitialBaseConditions<V>,
    friend: SampleInitialFriendConditions<V>,
    social: SampleInitialSocialConditions<V>,
}

impl<V> InitialConditions<V>
where
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    pub fn sample<R: Rng>(&self, rng: &mut R) -> SampleInitialConditions<V> {
        let base = self.base.sample(rng);
        let friend = self.friend.sample(rng, &base);
        let social = self.social.sample(rng);

        SampleInitialConditions {
            base,
            friend,
            social,
        }
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialBaseConditions<V>
where
    V: Float + UlpsEq + AddAssign,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    /// $H^F \implies A$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub a_fh: MArrD1<FH, SimplexDist<A, V>>,
    /// $H^K \implies B$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub b_kh: MArrD1<KH, SimplexDist<B, V>>,
    /// $B \implies O$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub o_b: MArrD1<B, SimplexDist<O, V>>,
    /// $H \implies \Theta$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub theta_h: MArrD1<H, SimplexDist<Theta, V>>,
    /// $H \implies \Theta'$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub thetad_h: MArrD1<H, SimplexDist<Thetad, V>>,
    /// parameters of conditional opinions $\Psi \implies H$ and $B \implies H$ when $\phi_0$ is true
    pub params_h_psi_b_if_phi0: ConditionParams<Psi, B, H, V>,
}

struct SampleInitialBaseConditions<V> {
    o_b: MArrD1<B, SimplexD1<O, V>>,
    b_kh: MArrD1<KH, SimplexD1<B, V>>,
    a_fh: MArrD1<FH, SimplexD1<A, V>>,
    theta_h: MArrD1<H, SimplexD1<Theta, V>>,
    thetad_h: MArrD1<H, SimplexD1<Thetad, V>>,
    h_psi_if_phi0: MArrD1<Psi, SimplexD1<H, V>>,
    h_b_if_phi0: MArrD1<B, SimplexD1<H, V>>,
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialFriendConditions<V>
where
    V: Float + UlpsEq + AddAssign,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    /// $M \implies \Psi^F$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub fpsi_m: MArrD1<M, SimplexDist<FPsi, V>>,
    /// $B^F \implies O^F$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub fo_fb: MArrD1<FB, SimplexDist<FO, V>>,
    /// $M^F \implies B^F$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub fb_fm: MArrD1<FM, SimplexDist<FB, V>>,
    /// parameters of conditional opinions $\Psi^F \implies H^F$ and $B^F \implies H^F$ when $\phi^F_0$ is true
    pub params_fh_fpsi_fb_if_fphi0: DependentParam<FH, V, ConditionParams<FPsi, FB, FH, V>>,
}

struct SampleInitialFriendConditions<V> {
    fpsi_m: MArrD1<M, SimplexD1<FPsi, V>>,
    fo_fb: MArrD1<FB, SimplexD1<FO, V>>,
    fb_fm: MArrD1<FM, SimplexD1<FB, V>>,
    fh_fpsi_if_fphi0: MArrD1<FPsi, SimplexD1<FH, V>>,
    fh_fb_if_fphi0: MArrD1<FB, SimplexD1<FH, V>>,
}

impl<V> InitialBaseConditions<V>
where
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SampleInitialBaseConditions<V> {
        let o_b = MArrD1::from_fn(|i| self.o_b[i].sample(rng));
        let b_kh = MArrD1::from_fn(|i| self.b_kh[i].sample(rng));
        let a_fh = MArrD1::from_fn(|i| self.a_fh[i].sample(rng));

        let theta_h = MArrD1::from_fn(|h| self.theta_h[h].sample(rng));
        let thetad_h = MArrD1::from_fn(|h| self.thetad_h[h].sample(rng));
        let (h_psi_if_phi0, h_b_if_phi0) = self.params_h_psi_b_if_phi0.sample(rng);

        debug!(target: "O||B", w =   ?o_b);
        debug!(target: "B||KTh", w = ?b_kh);
        debug!(target: "A||FTh", w = ?a_fh);
        debug!(target: "Th||...", w = ?theta_h);
        debug!(target: "Thd||..", w = ?thetad_h);
        debug!(target: "H||Psi,phi0", w = ?h_psi_if_phi0);
        debug!(target: "H||B,phi0",   w = ?h_b_if_phi0);

        SampleInitialBaseConditions {
            o_b,
            b_kh,
            a_fh,
            theta_h,
            thetad_h,
            h_psi_if_phi0,
            h_b_if_phi0,
        }
    }
}

impl<V> InitialFriendConditions<V>
where
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        base: &SampleInitialBaseConditions<V>,
    ) -> SampleInitialFriendConditions<V> {
        let fpsi_m = MArrD1::from_fn(|i| self.fpsi_m[i].sample(rng));
        let fb_fm = MArrD1::from_fn(|i| self.fb_fm[i].sample(rng));
        let fo_fb = MArrD1::from_fn(|i| self.fo_fb[i].sample(rng));
        let (fh_fpsi_if_fphi0, fh_fb_if_fphi0) = self
            .params_fh_fpsi_fb_if_fphi0
            .samples::<FPsi, FB, _, _, _>(
                rng,
                &(
                    MArrD1::<FPsi, _>::from_fn(|i| base.h_psi_if_phi0[i].clone().conv()),
                    MArrD1::<FB, _>::from_fn(|i| base.h_b_if_phi0[i].clone().conv()),
                ),
            );

        debug!(target: "FPsi||M", w = ?fpsi_m);
        debug!(target: "FB||FM", w = ?fb_fm);
        debug!(target: "FO||FB", w = ?fo_fb);
        debug!(target: "FH||FPsi,Fphi0", w = ?fh_fpsi_if_fphi0);
        debug!(target: "FH||FB,Fphi0", w = ?fh_fb_if_fphi0);

        SampleInitialFriendConditions {
            fpsi_m,
            fo_fb,
            fb_fm,
            fh_fpsi_if_fphi0,
            fh_fb_if_fphi0,
        }
    }
}

#[serde_as]
#[derive(Debug, serde::Deserialize)]
#[serde(bound(deserialize = "V: serde::Deserialize<'de>"))]
pub struct InitialSocialConditions<V>
where
    V: Float + UlpsEq + AddAssign,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    /// $M \implies \Psi^K$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub kpsi_m: MArrD1<M, SimplexDist<KPsi, V>>,
    /// $B^K \implies O^K$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub ko_kb: MArrD1<KB, SimplexDist<KO, V>>,
    /// $M^K \implies B^K$
    #[serde_as(as = "TryFromInto<Vec<SimplexParam<V>>>")]
    pub kb_km: MArrD1<KM, SimplexDist<KB, V>>,
    /// parameters of conditional opinions $\Psi^K \implies H^K$ and $B^K \implies H^K$ when $\phi^K_0$ is true
    pub params_kh_kpsi_kb_if_kphi0: ConditionParams<KPsi, KB, KH, V>,
}

pub struct SampleInitialSocialConditions<V> {
    kpsi_m: MArrD1<M, SimplexD1<KPsi, V>>,
    ko_kb: MArrD1<KB, SimplexD1<KO, V>>,
    kb_km: MArrD1<KM, SimplexD1<KB, V>>,
    kh_kpsi_if_kphi0: MArrD1<KPsi, SimplexD1<KH, V>>,
    kh_kb_if_kphi0: MArrD1<KB, SimplexD1<KH, V>>,
}

impl<V> InitialSocialConditions<V>
where
    V: MyFloat,
    Standard: Distribution<V>,
    StandardNormal: Distribution<V>,
    Exp1: Distribution<V>,
    Open01: Distribution<V>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SampleInitialSocialConditions<V> {
        let kpsi_m = MArrD1::from_fn(|i| self.kpsi_m[i].sample(rng));
        let ko_kb = MArrD1::from_fn(|i| self.ko_kb[i].sample(rng));
        let kb_km = MArrD1::from_fn(|i| self.kb_km[i].sample(rng));
        let (kh_kpsi_if_kphi0, kh_kb_if_kphi0) = self.params_kh_kpsi_kb_if_kphi0.sample(rng);

        debug!(target: "KPsi||M", w = ?kpsi_m);
        debug!(target: "KB||KM",  w = ?kb_km);
        debug!(target: "KO||KB",  w = ?ko_kb);
        debug!(target: "KH||KPsi,Kphi0", w = ?kh_kpsi_if_kphi0);
        debug!(target: "KH||KB,Kphi0", w = ?kh_kb_if_kphi0);

        SampleInitialSocialConditions {
            kb_km,
            ko_kb,
            kpsi_m,
            kh_kpsi_if_kphi0,
            kh_kb_if_kphi0,
        }
    }
}

#[derive(Debug, Default)]
pub struct ConditionalOpinions<V: Float> {
    base: BaseConditionalOpinions<V>,
    friend: FriendConditionalOpinions<V>,
    social: SocialConditionalOpinions<V>,
}

#[derive(Debug, Default)]
pub struct BaseConditionalOpinions<V: Float> {
    /// $B H^K \implies O$
    b_kh_o: MArrD2<KH, O, SimplexD1<B, V>>,
    /// $\Theta^K \implies A$
    a_fh: MArrD1<FH, SimplexD1<A, V>>,
    /// $H \implies \Theta$
    theta_h: MArrD1<H, SimplexD1<Theta, V>>,
    /// $H \implies \Theta'$
    thetad_h: MArrD1<H, SimplexD1<Thetad, V>>,
    /// $\Psi,B \implies H$ when $\phi_0$ is true
    h_psi_b_if_phi0: MArrD2<Psi, B, SimplexD1<H, V>>,
}

#[derive(Debug, Default)]
pub struct FriendConditionalOpinions<V: Float> {
    /// $S \implies \Psi^F$
    fpsi_m: MArrD1<M, SimplexD1<FPsi, V>>,
    /// $M^F,O^F \implies B^F$
    fb_fm_fo: MArrD2<FM, FO, SimplexD1<FB, V>>,
    /// $\Psi^F,B^F \implies H^F$ when $\phi^F_0$ is true
    fh_fpsi_fb_if_fpsi0: MArrD2<FPsi, FB, SimplexD1<FH, V>>,
}

#[derive(Debug, Default)]
pub struct SocialConditionalOpinions<V: Float> {
    /// $M \implies \Psi^K$
    kpsi_m: MArrD1<M, SimplexD1<KPsi, V>>,
    /// $M^K,O^K \implies B^K$
    kb_km_ko: MArrD2<KM, KO, SimplexD1<KB, V>>,
    /// $\Psi^K,B^K \implies H^K$ when $\phi^K_0$ is true
    kh_kpsi_kb_if_kpsi0: MArrD2<KPsi, KB, SimplexD1<KH, V>>,
}

#[derive(Debug, Default)]
pub struct Opinions<V: Float> {
    op: BaseOpinions<V>,
    sop: CollectiveOpinions<V>,
    fop: FriendOpinions<V>,
}

#[derive(Debug, Default)]
struct MyOpinions<V: Float> {
    op: BaseOpinions<V>,
    sop: CollectiveOpinions<V>,
    fop: FriendOpinions<V>,
    fb: OpinionD1<FB, V>,
    kb: OpinionD1<KB, V>,
    kh: OpinionD1<KH, V>,
    b: OpinionD1<B, V>,
    a: OpinionD1<A, V>,
    theta: OpinionD1<Theta, V>,
    thetad: OpinionD1<Thetad, V>,
}

#[derive(Debug, Default, Clone)]
struct BaseOpinions<V: Float> {
    psi: OpinionD1<Psi, V>,
    phi: OpinionD1<Phi, V>,
    m: OpinionD1<M, V>,
    o: OpinionD1<O, V>,
    h_psi_if_phi1: MArrD1<Psi, SimplexD1<H, V>>,
    h_b_if_phi1: MArrD1<B, SimplexD1<H, V>>,
}

#[derive(Debug, Default, Clone)]
pub struct CollectiveOpinions<V: Float> {
    ko: OpinionD1<KO, V>,
    km: OpinionD1<KM, V>,
    kphi: OpinionD1<KPhi, V>,
    kh_kpsi_if_kphi1: MArrD1<KPsi, SimplexD1<KH, V>>,
    kh_kb_if_kphi1: MArrD1<KB, SimplexD1<KH, V>>,
}

#[derive(Debug, Default)]
pub struct FriendOpinions<V: Float> {
    fo: OpinionD1<FO, V>,
    fm: OpinionD1<FM, V>,
    fphi: OpinionD1<FPhi, V>,
    fh_fpsi_if_fphi1: MArrD1<FPsi, SimplexD1<FH, V>>,
    fh_fb_if_fphi1: MArrD1<FB, SimplexD1<FH, V>>,
}

#[derive(Debug)]
pub struct TempOpinions<V: Float> {
    h: OpinionD1<H, V>,
    theta: OpinionD1<Theta, V>,
    fpsi_ded: OpinionD1<FPsi, V>,
}

impl<V: Float> TempOpinions<V> {
    #[inline]
    pub fn get_theta_projection(&self) -> MArrD1<Theta, V>
    where
        V: NumAssign,
    {
        self.theta.projection()
    }
}

impl<V: Float> BaseOpinions<V> {
    fn new(
        init: InitialBaseSimplexes<V>,
        base_rates: &BaseRates<V>,
        h_if_phi1_psi0: SimplexD1<H, V>,
        h_if_phi1_b0: SimplexD1<H, V>,
    ) -> Self
    where
        V: UlpsEq + AddAssign,
    {
        let InitialBaseSimplexes {
            psi,
            phi,
            m,
            o,
            h_if_phi1_psi1,
            h_if_phi1_b1,
        } = init;
        Self {
            psi: (psi, base_rates.psi.clone()).into(),
            phi: (phi, base_rates.phi.clone()).into(),
            m: (m, base_rates.m.clone()).into(),
            o: (o, base_rates.o.clone()).into(),
            h_psi_if_phi1: marr_d1![h_if_phi1_psi0, h_if_phi1_psi1],
            h_b_if_phi1: marr_d1![h_if_phi1_b0, h_if_phi1_b1],
        }
    }

    fn update(&self, info: InfoContent<'_, V>, trust: V) -> Self
    where
        V: UlpsEq + NumAssign + Sum,
    {
        let psi = FuseOp::Wgh.fuse(&self.psi, &info.psi.discount(trust));
        let phi = FuseOp::Wgh.fuse(&self.phi, &info.phi.discount(trust));
        let m = FuseOp::Wgh.fuse(&self.m, &info.m.discount(trust));
        let o = FuseOp::Wgh.fuse(&self.o, &info.o.discount(trust));
        let h_psi_if_phi1 = marr_d1![
            self.h_psi_if_phi1[0].clone(),
            FuseOp::Wgh.fuse(&self.h_psi_if_phi1[1], &info.h_if_phi1_psi1.discount(trust))
        ];
        let h_b_if_phi1 = marr_d1![
            self.h_b_if_phi1[0].clone(),
            FuseOp::Wgh.fuse(&self.h_b_if_phi1[1], &info.h_if_phi1_b1.discount(trust))
        ];

        Self {
            psi,
            phi,
            m,
            o,
            h_psi_if_phi1,
            h_b_if_phi1,
        }
    }

    fn compute_a(
        &self,
        fh: &OpinionD1<FH, V>,
        conds: &BaseConditionalOpinions<V>,
        base_rates: &BaseRates<V>,
    ) -> OpinionD1<A, V>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        fh.deduce(&conds.a_fh)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.a.clone()))
    }

    fn compute_b(
        &self,
        kh: &OpinionD1<KH, V>,
        conds: &BaseConditionalOpinions<V>,
        base_rates: &BaseRates<V>,
    ) -> OpinionD1<B, V>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        OpinionD2::product2(kh, &self.o)
            .deduce(&conds.b_kh_o)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.b.clone()))
    }

    fn compute_h(
        &self,
        b: &OpinionD1<B, V>,
        conds: &BaseConditionalOpinions<V>,
        base_rates: &BaseRates<V>,
    ) -> OpinionD1<H, V>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        let h_psi_b_if_phi1: MArrD2<Psi, B, SimplexD1<H, V>> = MArrD1::<H, _>::merge_cond2(
            &self.h_psi_if_phi1,
            &self.h_b_if_phi1,
            &base_rates.psi,
            &base_rates.b,
            &base_rates.h,
        )
        .unwrap_or_else(|| MArrD2::new(vec![self.h_b_if_phi1.clone(), self.h_b_if_phi1.clone()]));

        let cond_h =
            MArrD3::<Phi, _, _, _>::new(vec![conds.h_psi_b_if_phi0.clone(), h_psi_b_if_phi1]);

        OpinionD3::product3(self.phi.as_ref(), b.as_ref(), self.psi.as_ref())
            .deduce(&cond_h)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.h.clone()))
    }

    fn compute_theta(
        &self,
        h: &OpinionD1<H, V>,
        conds: &BaseConditionalOpinions<V>,
        base_rates: &BaseRates<V>,
    ) -> OpinionD1<Theta, V>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        h.deduce(&conds.theta_h)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.theta.clone()))
    }

    fn compute_thetad(
        &self,
        h: &OpinionD1<H, V>,
        conds: &BaseConditionalOpinions<V>,
        base_rates: &BaseRates<V>,
    ) -> OpinionD1<Thetad, V>
    where
        V: Float + UlpsEq + NumAssign + Sum + Default,
    {
        h.deduce(&conds.thetad_h)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.thetad.clone()))
    }
}

impl<V: Float> CollectiveOpinions<V> {
    fn new(
        init: InitialSocialSimplexes<V>,
        base_rates: &SocialBaseRates<V>,
        kh_if_kphi1_kpsi0: SimplexD1<KH, V>,
        kh_if_kphi1_kb0: SimplexD1<KH, V>,
    ) -> Self
    where
        V: UlpsEq + AddAssign,
    {
        let InitialSocialSimplexes {
            kphi,
            km,
            ko,
            kh_if_kphi1_kb1,
            kh_if_kphi1_kpsi1,
        } = init;
        Self {
            kphi: (kphi, base_rates.kphi.clone()).into(),
            km: (km, base_rates.km.clone()).into(),
            ko: (ko, base_rates.ko.clone()).into(),
            kh_kpsi_if_kphi1: marr_d1![kh_if_kphi1_kpsi0, kh_if_kphi1_kpsi1],
            kh_kb_if_kphi1: marr_d1![kh_if_kphi1_kb0, kh_if_kphi1_kb1],
        }
    }

    fn update(&self, info: InfoContent<'_, V>, trust: V) -> Self
    where
        V: UlpsEq + NumAssign + Sum,
    {
        // compute friends opinions
        let ko = FuseOp::Wgh.fuse(&self.ko, &info.o.discount(trust).conv());
        let km = FuseOp::Wgh.fuse(&self.km, &info.m.discount(trust).conv());
        let kphi = FuseOp::Wgh.fuse(&self.kphi, &info.phi.discount(trust).conv());
        let kh_kpsi_if_kphi1 = marr_d1![
            self.kh_kpsi_if_kphi1[0].clone(),
            FuseOp::Wgh.fuse(
                &self.kh_kpsi_if_kphi1[1],
                &info.h_if_phi1_psi1.discount(trust).conv()
            ),
        ];
        let kh_kb_if_kphi1 = marr_d1![
            self.kh_kb_if_kphi1[0].clone(),
            FuseOp::Wgh.fuse(
                &self.kh_kb_if_kphi1[1],
                &info.h_if_phi1_b1.discount(trust).conv()
            ),
        ];

        Self {
            ko,
            km,
            kphi,
            kh_kpsi_if_kphi1,
            kh_kb_if_kphi1,
        }
    }

    fn compute_kh(
        &self,
        kb: &OpinionD1<KB, V>,
        kpsi: &OpinionD1<KPsi, V>,
        conds: &SocialConditionalOpinions<V>,
        base_rates: &SocialBaseRates<V>,
    ) -> OpinionD1<KH, V>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        let kh_kpsi_kb_if_kpsi1: MArrD2<KPsi, KB, _> = MArrD1::<KH, _>::merge_cond2(
            &self.kh_kpsi_if_kphi1,
            &self.kh_kb_if_kphi1,
            &base_rates.kpsi,
            &base_rates.kb,
            &base_rates.kh,
        )
        .unwrap_or_else(|| {
            MArrD2::new(vec![
                self.kh_kb_if_kphi1.clone(),
                self.kh_kb_if_kphi1.clone(),
            ])
        });
        let cond_kh = MArrD3::<KPhi, _, _, _>::new(vec![
            conds.kh_kpsi_kb_if_kpsi0.clone(),
            kh_kpsi_kb_if_kpsi1,
        ]);
        let mw = OpinionD3::product3(&self.kphi, kb, kpsi);
        mw.deduce(&cond_kh)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.kh.clone()))
    }

    fn compute_kb(
        &self,
        conds: &SocialConditionalOpinions<V>,
        base_rates: &SocialBaseRates<V>,
    ) -> OpinionD1<KB, V>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        OpinionD2::product2(&self.km, &self.ko)
            .deduce(&conds.kb_km_ko)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.kb.clone()))
    }

    fn compute_kpsi(
        info_kpsi: &SimplexD1<KPsi, V>,
        m: &OpinionD1<M, V>,
        trust: V,
        conds: &SocialConditionalOpinions<V>,
        base_rates: &SocialBaseRates<V>,
    ) -> OpinionD1<KPsi, V>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        let mut kpsi = m
            .deduce(&conds.kpsi_m)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.kpsi.clone()));
        FuseOp::Wgh.fuse_assign(&mut kpsi, &info_kpsi.discount(trust));
        kpsi
    }
}

impl<V: Float> FriendOpinions<V> {
    fn new(
        init: InitialFriendSimplexes<V>,
        base_rates: &FriendBaseRates<V>,
        fh_if_fphi1_fpsi0: SimplexD1<FH, V>,
        fh_if_fphi1_fb0: SimplexD1<FH, V>,
    ) -> Self
    where
        V: UlpsEq + AddAssign,
    {
        let InitialFriendSimplexes {
            fphi,
            fm,
            fo,
            fh_if_fphi1_fpsi1,
            fh_if_fphi1_fb1,
        } = init;
        Self {
            fphi: (fphi, base_rates.fphi.clone()).into(),
            fm: (fm, base_rates.fm.clone()).into(),
            fo: (fo, base_rates.fo.clone()).into(),
            fh_fpsi_if_fphi1: marr_d1![fh_if_fphi1_fpsi0, fh_if_fphi1_fpsi1],
            fh_fb_if_fphi1: marr_d1![fh_if_fphi1_fb0, fh_if_fphi1_fb1],
        }
    }

    fn update(&self, info: InfoContent<'_, V>, trust: V) -> Self
    where
        V: Float + UlpsEq + NumAssign + Sum + fmt::Debug,
    {
        // compute friends opinions
        let fo = FuseOp::Wgh.fuse(&self.fo, &info.o.discount(trust).conv());
        let fm = FuseOp::Wgh.fuse(&self.fm, &info.m.discount(trust).conv());
        let fphi = FuseOp::Wgh.fuse(&self.fphi, &info.phi.discount(trust).conv());
        let fh_fpsi_if_fphi1 = marr_d1![
            self.fh_fpsi_if_fphi1[0].clone(),
            FuseOp::Wgh.fuse(
                &self.fh_fpsi_if_fphi1[1],
                &info.h_if_phi1_psi1.discount(trust).conv(),
            )
        ];
        let fh_fb_if_fphi1 = marr_d1![
            self.fh_fb_if_fphi1[0].clone(),
            FuseOp::Wgh.fuse(
                &self.fh_fb_if_fphi1[1],
                &info.h_if_phi1_b1.discount(trust).conv()
            )
        ];

        debug!(target: "    FM", w = ?fm);

        Self {
            fo,
            fm,
            fphi,
            fh_fpsi_if_fphi1,
            fh_fb_if_fphi1,
        }
    }

    fn deduce_fpsi(
        m: &OpinionD1<M, V>,
        conds: &FriendConditionalOpinions<V>,
        base_rates: &FriendBaseRates<V>,
    ) -> OpinionD1<FPsi, V>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        m.deduce(&conds.fpsi_m)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.fpsi.clone()))
    }

    fn compute_fb(
        &self,
        conds: &FriendConditionalOpinions<V>,
        base_rates: &FriendBaseRates<V>,
    ) -> OpinionD1<FB, V>
    where
        V: UlpsEq + NumAssign + Sum + Default + fmt::Debug,
    {
        OpinionD2::product2(&self.fm, &self.fo)
            .deduce(&conds.fb_fm_fo)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.fb.clone()))
    }

    fn compute_fpsi(
        info_fpsi: &SimplexD1<FPsi, V>,
        fpsi_ded: &OpinionD1<FPsi, V>,
        trust: V,
    ) -> OpinionD1<FPsi, V>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        FuseOp::Wgh.fuse(fpsi_ded, &info_fpsi.discount(trust))
    }

    fn compute_fh(
        &self,
        fb: &OpinionD1<FB, V>,
        fpsi: &OpinionD1<FPsi, V>,
        conds: &FriendConditionalOpinions<V>,
        base_rates: &FriendBaseRates<V>,
    ) -> OpinionD1<FH, V>
    where
        V: UlpsEq + NumAssign + Sum + Default,
    {
        let fh_fpsi_fb_fpsi1: MArrD2<FPsi, FB, _> = MArrD1::<FH, _>::merge_cond2(
            &self.fh_fpsi_if_fphi1,
            &self.fh_fb_if_fphi1,
            &base_rates.fpsi,
            &base_rates.fb,
            &base_rates.fh,
        )
        .unwrap_or_else(|| {
            MArrD2::new(vec![
                self.fh_fb_if_fphi1.clone(),
                self.fh_fb_if_fphi1.clone(),
            ])
        });
        let cond_fh =
            MArrD3::<FPhi, _, _, _>::new(vec![conds.fh_fpsi_fb_if_fpsi0.clone(), fh_fpsi_fb_fpsi1]);
        let mw = OpinionD3::product3(&self.fphi, fb, fpsi);
        mw.deduce(&cond_fh)
            .unwrap_or_else(|| OpinionD1::vacuous_with(base_rates.fh.clone()))
    }

    fn compute_p_a_thetad(
        &self,
        op: &BaseOpinions<V>,
        temp: &TempOpinions<V>,
        info_fpsi: &SimplexD1<FPsi, V>,
        friend_trust: V,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> MArrD2<A, Thetad, V>
    where
        V: UlpsEq + NumAssign + Sum + Default + fmt::Debug,
    {
        let fb = self.compute_fb(&conds.friend, &base_rates.friend);
        debug!(target: "    FB", w = ?fb);
        let fpsi = FriendOpinions::compute_fpsi(info_fpsi, &temp.fpsi_ded, friend_trust);
        debug!(target: "  FPsi", w = ?fpsi);
        let fh = self.compute_fh(&fb, &fpsi, &conds.friend, &base_rates.friend);
        debug!(target: "    FH", w = ?fh);
        let a = op.compute_a(&fh, &conds.base, &base_rates.base);
        debug!(target: "     A", w = ?a);
        let thetad = op.compute_thetad(&temp.h, &conds.base, &base_rates.base);
        debug!(target: "   Thd", w = ?thetad);

        let a_thetad = OpinionD2::product2(a.as_ref(), thetad.as_ref());
        let p = a_thetad.projection();
        debug!(target: " A,Thd", P = ?p);
        p
    }
}

impl<V: Float> Opinions<V> {
    pub fn new(
        initial_opinions: InitialOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
        sample: &SampleInitialConditions<V>,
    ) -> Self
    where
        V: UlpsEq + NumAssign,
    {
        let InitialOpinions {
            base,
            friend,
            social,
        } = initial_opinions;

        let op = BaseOpinions::new(
            base,
            &base_rates.base,
            sample.base.h_psi_if_phi0[0].clone(),
            sample.base.h_b_if_phi0[0].clone(),
        );
        let fop = FriendOpinions::new(
            friend,
            &base_rates.friend,
            sample.friend.fh_fpsi_if_fphi0[0].clone(),
            sample.friend.fh_fb_if_fphi0[0].clone(),
        );
        let sop = CollectiveOpinions::new(
            social,
            &base_rates.social,
            sample.social.kh_kpsi_if_kphi0[0].clone(),
            sample.social.kh_kb_if_kphi0[0].clone(),
        );

        Self { op, fop, sop }
    }

    pub fn receive_info(
        self,
        info: InfoContent<'_, V>,
        trust: V,
        friend_trust: V,
        social_trust: V,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> MyOpinions<V>
    where
        V: MyFloat,
    {
        let Opinions { op, sop, fop } = self.update(info, trust, friend_trust, social_trust);
        let kb = sop.compute_kb(&conds.social, &base_rates.social);
        let kpsi = CollectiveOpinions::compute_kpsi(
            &info.psi.to_owned().conv(),
            &op.m,
            social_trust,
            &conds.social,
            &base_rates.social,
        );
        let kh = sop.compute_kh(&kb, &kpsi, &conds.social, &base_rates.social);
        let b = op.compute_b(&kh, &conds.base, &base_rates.base);
        let h = op.compute_h(&b, &conds.base, &base_rates.base);
        let theta = op.compute_theta(&h, &conds.base, &base_rates.base);
        let fb = fop.compute_fb(&conds.friend, &base_rates.friend);
        let info_fpsi = info.psi.to_owned().conv();
        let fpsi_ded = FriendOpinions::deduce_fpsi(&op.m, &conds.friend, &base_rates.friend);
        let fpsi = FriendOpinions::compute_fpsi(&info_fpsi, &fpsi_ded, friend_trust);
        let fh = fop.compute_fh(&fb, &fpsi, &conds.friend, &base_rates.friend);
        let a = op.compute_a(&fh, &conds.base, &base_rates.base);
        let thetad = op.compute_thetad(&h, &conds.base, &base_rates.base);

        MyOpinions {
            op,
            sop,
            fop,
            fb,
            kb,
            kh,
            b,
            a,
            theta,
            thetad,
        }
    }

    pub fn predict2(
        &self,
        info: InfoContent<'_, V>,
        friend_trust: V,
        pred_friend_trust: V,
        social_trust: V,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> MyOpinions<V>
    where
        V: MyFloat,
    {
        let temp_ops = self.compute(info, social_trust, conds, &base_rates);
        let (pred_new_fop, _) = self.predict(
            &temp_ops,
            info,
            friend_trust,
            pred_friend_trust,
            conds,
            base_rates,
        );
        MyOpinions {
            sop: self.sop.clone(),
            op: self.op.clone(),
            fop: pred_new_fop,
            fb: todo!(),
            kb: todo!(),
            kh: todo!(),
            b: todo!(),
            a: todo!(),
            theta: todo!(),
            thetad: todo!(),
        }
    }

    pub fn update(
        &self,
        info: InfoContent<'_, V>,
        trust: V,
        friend_trust: V,
        social_trust: V,
    ) -> Self
    where
        V: UlpsEq + NumAssign + Sum + fmt::Debug,
    {
        let op = self.op.update(info, trust);
        let sop = self.sop.update(info, social_trust);
        let fop = self.fop.update(info, friend_trust);

        Self { op, sop, fop }
    }

    pub fn compute(
        &self,
        info: InfoContent<V>,
        social_trust: V,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> TempOpinions<V>
    where
        V: UlpsEq + NumAssign + Sum + Default + std::fmt::Debug,
    {
        let fpsi_ded = FriendOpinions::deduce_fpsi(&self.op.m, &conds.friend, &base_rates.friend);
        let kb = self.sop.compute_kb(&conds.social, &base_rates.social);
        let kpsi = CollectiveOpinions::compute_kpsi(
            &info.psi.to_owned().conv(),
            &self.op.m,
            social_trust,
            &conds.social,
            &base_rates.social,
        );
        let kh = self
            .sop
            .compute_kh(&kb, &kpsi, &conds.social, &base_rates.social);
        let b = self.op.compute_b(&kh, &conds.base, &base_rates.base);
        let h = self.op.compute_h(&b, &conds.base, &base_rates.base);
        let theta = self.op.compute_theta(&h, &conds.base, &base_rates.base);

        debug!(target: "     S", w = ?self.op.m);
        debug!(target: "   Psi", w = ?self.op.psi);
        debug!(target: " FPsid", w = ?fpsi_ded);
        debug!(target: "    KB", w = ?kb);
        debug!(target: "  KPsi", w = ?kpsi);
        debug!(target: "    KH", w = ?kh);
        debug!(target: "     B", w = ?b);
        debug!(target: "     H", w = ?h);
        debug!(target: "    Th", w = ?theta);

        TempOpinions { theta, h, fpsi_ded }
    }

    pub fn predict(
        &self,
        temp: &TempOpinions<V>,
        info: InfoContent<'_, V>,
        friend_trust: V,
        pred_friend_trust: V,
        conds: &ConditionalOpinions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> (FriendOpinions<V>, [MArrD2<A, Thetad, V>; 2])
    where
        V: UlpsEq + Sum + Default + NumAssign + fmt::Debug,
    {
        // current opinions
        let span = span!(Level::DEBUG, "pred-curr");
        let _guard = span.enter();
        let info_fpsi = info.psi.to_owned().conv();
        let p_a_thetad = self.fop.compute_p_a_thetad(
            &self.op,
            temp,
            &info_fpsi,
            friend_trust,
            conds,
            base_rates,
        );
        drop(_guard);

        // predict  opinions in case of sharing
        let span = span!(Level::DEBUG, "pred-pred");
        let _guard = span.enter();
        let pred_fop = self.fop.update(info, pred_friend_trust);
        let p_pred_a_thetad = pred_fop.compute_p_a_thetad(
            &self.op,
            temp,
            &info_fpsi,
            pred_friend_trust,
            conds,
            base_rates,
        );
        drop(_guard);

        let ps = [p_a_thetad, p_pred_a_thetad];
        (pred_fop, ps)
    }

    pub fn replace_pred_fop(&mut self, pred_fop: FriendOpinions<V>) {
        self.fop = pred_fop;
    }
}

pub trait MyFloat
where
    Self: Float
        + NumAssign
        + UlpsEq
        + fmt::Debug
        + Sum
        + FromPrimitive
        + SampleUniform
        + ToPrimitive
        + Default
        + Send
        + Sync,
{
}

impl MyFloat for f32 {}
impl MyFloat for f64 {}

impl<V: Float> ConditionalOpinions<V> {
    pub fn from_sample(sample: SampleInitialConditions<V>, base_rates: &GlobalBaseRates<V>) -> Self
    where
        V: MyFloat,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        let base = BaseConditionalOpinions::from_sample(sample.base, &base_rates);
        let friend = FriendConditionalOpinions::from_sample(sample.friend, &base_rates.friend);
        let social = SocialConditionalOpinions::from_sample(sample.social, &base_rates.social);

        Self {
            base,
            friend,
            social,
        }
    }
}

impl<V> BaseConditionalOpinions<V>
where
    V: MyFloat,
{
    fn from_sample(
        sample: SampleInitialBaseConditions<V>,
        base_rates: &GlobalBaseRates<V>,
    ) -> Self {
        let SampleInitialBaseConditions {
            o_b,
            b_kh,
            a_fh,
            theta_h,
            thetad_h,
            h_psi_if_phi0,
            h_b_if_phi0,
        } = sample;

        let h_psi_b_if_phi0 = MArrD1::<H, _>::merge_cond2(
            &h_psi_if_phi0,
            &h_b_if_phi0,
            &base_rates.base.psi,
            &base_rates.base.b,
            &base_rates.base.h,
        )
        .expect("failed to merge into Psi,B,phi0=>H");

        let b_o = o_b
            .inverse(&base_rates.base.b, &base_rates.base.o)
            .expect("failed to invert B=>O");
        let b_kh_o = MArrD1::<B, _>::merge_cond2(
            &b_kh,
            &b_o,
            &base_rates.social.kh,
            &base_rates.base.o,
            &base_rates.base.b,
        )
        .expect("failed to merge into KH,O=>B");

        Self {
            a_fh,
            b_kh_o,
            theta_h,
            thetad_h,
            h_psi_b_if_phi0,
        }
    }
}

impl<V: MyFloat> FriendConditionalOpinions<V> {
    fn from_sample(
        sample: SampleInitialFriendConditions<V>,
        base_rates: &FriendBaseRates<V>,
    ) -> Self
    where
        V: MyFloat,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        let SampleInitialFriendConditions {
            fpsi_m,
            fo_fb,
            fb_fm,
            fh_fpsi_if_fphi0,
            fh_fb_if_fphi0,
        } = sample;

        let fh_fpsi_fb_if_fpsi0 = MArrD1::<FH, _>::merge_cond2(
            &fh_fpsi_if_fphi0,
            &fh_fb_if_fphi0,
            &base_rates.fpsi,
            &base_rates.fb,
            &base_rates.fh,
        )
        .expect("failed to merge into FPsi,FB,Fpsi0=>FH");

        let fb_fo = fo_fb
            .inverse(&base_rates.fb, &base_rates.fo)
            .expect("failed to invert FB=>FO");
        let fb_fm_fo = MArrD1::<FB, _>::merge_cond2(
            &fb_fm,
            &fb_fo,
            &base_rates.fm,
            &base_rates.fo,
            &base_rates.fb,
        )
        .expect("failed to merge into FM,FO=>FB");

        Self {
            fb_fm_fo,
            fpsi_m,
            fh_fpsi_fb_if_fpsi0,
        }
    }
}

impl<V: MyFloat> SocialConditionalOpinions<V> {
    pub fn from_sample(
        sample: SampleInitialSocialConditions<V>,
        base_rates: &SocialBaseRates<V>,
    ) -> Self
    where
        V: MyFloat,
        Standard: Distribution<V>,
        StandardNormal: Distribution<V>,
        Exp1: Distribution<V>,
        Open01: Distribution<V>,
    {
        let SampleInitialSocialConditions {
            kpsi_m,
            ko_kb,
            kb_km,
            kh_kpsi_if_kphi0,
            kh_kb_if_kphi0,
        } = sample;

        let kh_kpsi_kb_if_kpsi0 = MArrD1::<KH, _>::merge_cond2(
            &kh_kpsi_if_kphi0,
            &kh_kb_if_kphi0,
            &base_rates.kpsi,
            &base_rates.kb,
            &base_rates.kh,
        )
        .expect("failed to merge into KPsi,KB,Kpsi0=>KH");

        let kb_ko = ko_kb
            .inverse(&base_rates.kb, &base_rates.ko)
            .expect("failed to invert KB=>KO");
        let kb_km_ko = MArrD1::<KB, _>::merge_cond2(
            &kb_km,
            &kb_ko,
            &base_rates.km,
            &base_rates.ko,
            &base_rates.kb,
        )
        .expect("failed to merge into KM,KO=>KB");

        Self {
            kpsi_m,
            kb_km_ko,
            kh_kpsi_kb_if_kpsi0,
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use rand_distr::Distribution;
    use subjective_logic::iter::FromFn;
    use subjective_logic::mul::labeled::SimplexD1;
    use subjective_logic::multi_array::labeled::MArrD1;
    use subjective_logic::{approx_ext, marr_d1};
    use subjective_logic::{domain::Domain, impl_domain};

    use super::{SimplexDist, SimplexParam};

    struct X;
    impl_domain!(X = 2);

    struct Y;
    impl_domain!(Y = 3);

    #[test]
    fn test_simplex_dist_conversion() {
        let mut rng = thread_rng();
        let b = vec![0.1f32, 0.2];
        let u = 0.7;
        let fp = SimplexParam::Fixed(b.clone(), u);
        let d = SimplexDist::<X, _>::try_from(fp).unwrap();
        let s = d.sample(&mut rng);
        assert_eq!(s.belief, MArrD1::try_from(b).unwrap());
        assert_eq!(s.uncertainty, u);

        let edp = SimplexParam::Dirichlet {
            alpha: vec![5.0f32, 5.0, 1.0],
            zeros: None,
        };
        let ed = SimplexDist::<Y, _>::try_from(edp);
        assert!(ed.is_err());
        if let Err(e) = ed {
            println!("{:}", e);
        }
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0, 1.0],
            zeros: None,
        };
        let d = SimplexDist::<X, _>::try_from(dp);
        let n = 25;
        assert!(d.is_ok());
        let ss = d.unwrap().sample_iter(&mut rng).take(n).collect::<Vec<_>>();
        let mean_b0 = ss.iter().map(|s| s.b()[0]).sum::<f32>() / n as f32;
        let mean_b1 = ss.iter().map(|s| s.b()[1]).sum::<f32>() / n as f32;
        let mean_u = ss.iter().map(|s| s.u()).sum::<f32>() / n as f32;
        println!("{mean_b0:?}, {mean_b1:?}, {mean_u:?}");
        assert!(mean_b0 > mean_b1 && mean_b1 > mean_u);
    }

    #[test]
    fn test_simplex_zeros() {
        let rng = &mut thread_rng();

        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0],
            zeros: Some(vec![2]),
        };
        let d = SimplexDist::<X, _>::try_from(dp);
        assert!(d.is_ok());
        let s = d.unwrap().sample(rng);
        assert_eq!(*s.u(), 0.0f32);
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0],
            zeros: Some(vec![0]),
        };
        let d = SimplexDist::<X, _>::try_from(dp);
        assert!(d.is_ok());
        let s = d.unwrap().sample(rng);
        assert_eq!(s.b()[0], 0.0f32);
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0],
            zeros: Some(vec![1]),
        };
        let d = SimplexDist::<X, _>::try_from(dp);
        assert!(d.is_ok());
        let s = d.unwrap().sample(rng);
        assert_eq!(s.b()[1], 0.0f32);

        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0, 1.0],
            zeros: Some(vec![1]),
        };
        let d = SimplexDist::<X, _>::try_from(dp);
        assert!(d.is_err());
        println!("{}", d.err().unwrap());
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0],
            zeros: Some(vec![1, 1]),
        };
        let d = SimplexDist::<Y, _>::try_from(dp);
        assert!(d.is_err());
        println!("{}", d.err().unwrap());
        let dp = SimplexParam::Dirichlet {
            alpha: vec![7.0, 4.0],
            zeros: Some(vec![3]),
        };
        let d = SimplexDist::<X, _>::try_from(dp);
        assert!(d.is_err());
        println!("{}", d.err().unwrap());
    }

    /*
    #[test]
    fn test_relative_param() {
        let w = SimplexD1::<X, _>::new(marr_d1![0.9, 0.1], 0.00);
        let wd = RelativeParam {
            belief: 1.0,
            uncertainty: 1.0,
        }
        .to_simplex(&w)
        .unwrap();
        assert_eq!(w, wd);

        let w = SimplexD1::<X, _>::new(marr_d1![0.0, 0.90], 0.10);
        let wd = RelativeParam {
            belief: 1.0,
            uncertainty: 1.0,
        }
        .to_simplex(&w)
        .unwrap();
        assert_eq!(w, wd);

        let w = SimplexD1::<X, _>::new(marr_d1![0.0, 0.90], 0.10);
        let wd = RelativeParam {
            belief: 1.0,
            uncertainty: 0.0,
        }
        .to_simplex(&w)
        .unwrap();
        assert_eq!(wd.b()[1], 1.0);

        let w = SimplexD1::<X, _>::new(marr_d1![0.2, 0.30], 0.50);
        let wd = RelativeParam {
            belief: 1.0,
            uncertainty: 0.0,
        }
        .to_simplex(&w)
        .unwrap();
        assert_eq!(wd.b()[0], 0.4);
        assert_eq!(wd.b()[1], 0.6);

        let w = SimplexD1::<X, _>::new(marr_d1![0.2, 0.30], 0.50);
        let wd = RelativeParam {
            belief: 0.0,
            uncertainty: 1.2,
        }
        .to_simplex(&w)
        .unwrap();
        assert_eq!(wd.b()[0], 0.4);
        assert_eq!(*wd.u(), 0.6);
    }

    #[test]
    fn test_relative_vacuous() {
        let w0 = SimplexD1::<X, _>::new(marr_d1![0.0, 0.0], 1.0);
        let w1 = SimplexD1::<X, _>::new(marr_d1![0.9, 0.0], 0.1);
        let r = RelativeParam {
            belief: 1.0,
            uncertainty: 1.0,
        };
        let _rw0 = r.to_simplex(&w0).unwrap();
        let _rw1 = r.to_simplex(&w1).unwrap();
    }
    w*/

    /*
    #[test]
    fn test_cond_theta() -> anyhow::Result<()> {
        let mut rng = thread_rng();
        let cond_dist = toml::from_str::<CondThetaDist<f32>>(
            r#"
            b0psi0 = { Fixed = [[0.95, 0.00], 0.05] }
            b1psi1 = { Dirichlet = { alpha = [10.5, 10.5, 9.0]} }
            b0psi1 = { belief = { base = 1.2 }, uncertainty = { base = 1.0 } }
            b1psi0 = { belief = { base = 1.5 }, uncertainty = { base = 1.0 } }
        "#,
        )?;

        let rs = [
            cond_dist.b0psi1.sample(&mut rng),
            cond_dist.b1psi0.sample(&mut rng),
        ];

        for cond in cond_dist.sample_iter(&mut rng).take(10) {
            let w = &cond[(1, 1)];
            let wds = [&cond[(0, 1)], &cond[(1, 0)]];
            let x = w.b()[1] / (1.0 - *w.u());
            for (wd, r) in wds.into_iter().zip(&rs) {
                let xd = wd.b()[1] / (1.0 - *wd.u());
                assert!(r.belief > 1.0 / x || ulps_eq!(x * r.belief, xd));
            }
        }
        Ok(())
    } */

    /*
    #[test]
    fn test_base_cond() -> anyhow::Result<()> {
        let s = r#"
            psi  = [[0.0, 0.0], 1.0]
            phi  = [[0.0, 0.0], 1.0]
            s    = [[0.0, 0.0], 1.0]
            o    = [[0.0, 0.0], 1.0]
            a_fh = [
                { Fixed = [[0.95, 0.00], 0.05] },
                { Dirichlet = { alpha = [5.0, 5.0, 5.0]} },
            ]
            b_kh = [
                { Fixed = [[0.90, 0.00], 0.10] },
                { Dirichlet = { alpha = [5.0, 5.0, 5.0]} },
            ]
            o_b = [
                { Fixed = [[1.0, 0.00], 0.00] },
                { Dirichlet = { alpha = [5.0, 5.0, 5.0]} },
            ]
            h_phi = [
                { Fixed = [[0.00, 0.0], 1.00] },
                { Dirichlet = { alpha = [5.0, 5.0, 5.0]} },
            ]
            [cond_theta]
            b0psi0 = { Fixed = [[0.95, 0.00], 0.05] }
            b1psi1 = { Dirichlet = { alpha = [5.0, 5.0, 1.0]} }
            b0psi1 = { belief = { base = 1.0 }, uncertainty = { base = 1.0, error = {dist = "Standard", low = -0.2, high = 0.2 } } }
            b1psi0 = { belief = { base = 1.0 }, uncertainty = { base = 1.0, error = {dist = "Standard", low = -0.2, high = 0.2 } } }
            [rel_cond_thetad]
            belief = { base = 1.1 }
            uncertainty = { base = 1.1 }
        "#;

        let mut rng = thread_rng();

        let init_base_cond = toml::from_str::<InitialBaseConditions<f32>>(s)?;

        let r = init_base_cond.rel_cond_thetad.sample(&mut rng);
        for _ in 0..10 {
            let base_cond = BaseConditionalOpinions::from_init(&init_base_cond, &mut rng);
            for (k, wd) in base_cond.cond_thetad.iter_with() {
                // let wd = &[k];
                let w = match k {
                    (0, i, j) => &base_cond.cond_theta[(i, j)],
                    _ => &base_cond.cond_theta[(0, 0)],
                };
                let xd = wd.b()[1] / (1.0 - *wd.u());
                let x = w.b()[1] / (1.0 - *w.u());
                assert!(r.belief > 1.0 / x || ulps_eq!(x * r.belief, xd));
            }
            for (w, wd) in base_cond
                .cond_theta_phi
                .iter()
                .zip(&base_cond.cond_thetad_phi)
            {
                if w.is_vacuous() {
                    assert!(wd.is_vacuous());
                } else {
                    let xd = wd.b()[1] / (1.0 - *wd.u());
                    let x = w.b()[1] / (1.0 - *w.u());
                    assert!(r.belief > 1.0 / x || ulps_eq!(x * r.belief, xd));
                }
            }
        }
        Ok(())
    } */

    #[test]
    fn test_dirichlet_sampling() -> anyhow::Result<()> {
        let w = SimplexD1::<X, f64>::new(marr_d1![0.2, 0.3], 0.5);

        let s = 10.0;
        let d = marr_d1!(X; [0.5, 1.0]);
        let d_k = 1.0;
        let e = 0.0001;

        for d in d.iter().chain([&d_k]) {
            assert!(*d < 1.0 || approx_ext::is_one(*d));
        }

        let k = X::LEN as f64 + 1.0;
        let (alpha, alpha_k) = 'a: {
            let alpha = MArrD1::<X, _>::from_fn(|x| w.b()[x] * d[x] * s);
            let alpha_k = w.u() * d_k * s;
            let iter = alpha.iter().chain([&alpha_k]);
            for a in iter {
                if approx_ext::is_zero(*a) {
                    let s = alpha.iter().sum::<f64>() + alpha_k;
                    break 'a (
                        MArrD1::<X, _>::from_fn(|x| (alpha[x] + e) * s / (s + e * k)),
                        alpha_k * s / (s + e * k),
                    );
                }
            }
            (alpha, alpha_k)
        };

        println!("{:?}", alpha);
        println!("{:?}", alpha_k);
        println!("sum {:?}", alpha_k + alpha.iter().sum::<f64>());

        Ok(())
    }
}
