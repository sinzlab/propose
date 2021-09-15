from .base import BasePose, PoseSet


class Rat7mPose(BasePose):
    def __init__(self,
                 HeadF,
                 HeadB,
                 HeadL,
                 SpineF,
                 SpineM,
                 SpineL,
                 HipL,
                 KneeL,
                 ShinL,
                 HipR,
                 KneeR,
                 ShinR,
                 ElbowL,
                 ArmL,
                 ShoulderL,
                 ElbowR,
                 ArmR,
                 ShoulderR,
                 Offset1,
                 Offset2,
                 **kwargs
                 ):
        self.HeadF = HeadF
        self.HeadB = HeadB
        self.HeadL = HeadL
        self.SpineF = SpineF
        self.SpineM = SpineM
        self.SpineL = SpineL
        self.HipL = HipL
        self.KneeL = KneeL
        self.ShinL = ShinL
        self.HipR = HipR
        self.KneeR = KneeR
        self.ShinR = ShinR
        self.ElbowL = ElbowL
        self.ArmL = ArmL
        self.ShoulderL = ShoulderL
        self.ElbowR = ElbowR
        self.ArmR = ArmR
        self.ShoulderR = ShoulderR
        self.Offset1 = Offset1
        self.Offset2 = Offset2

        super().__init__(
            HeadF=HeadF,
            HeadB=HeadB,
            HeadL=HeadL,
            SpineF=SpineF,
            SpineM=SpineM,
            SpineL=SpineL,
            HipL=HipL,
            KneeL=KneeL,
            ShinL=ShinL,
            HipR=HipR,
            KneeR=KneeR,
            ShinR=ShinR,
            ElbowL=ElbowL,
            ArmL=ArmL,
            ShoulderL=ShoulderL,
            ElbowR=ElbowR,
            ArmR=ArmR,
            ShoulderR=ShoulderR,
            Offset1=Offset1,
            Offset2=Offset2,
        )