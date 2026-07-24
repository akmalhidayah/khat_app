"""Per-class softmax probabilities (probabilitas_prediksi)."""

from .database import db


class ProbabilitasPrediksi(db.Model):
    __tablename__ = "probabilitas_prediksi"

    id = db.Column(db.Integer, primary_key=True)
    id_riwayat = db.Column(
        db.Integer,
        db.ForeignKey("riwayat_klasifikasi.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    id_kelas = db.Column(db.Integer, db.ForeignKey("kelas_khat.id"), nullable=False, index=True)
    skor_softmax = db.Column(db.Float, nullable=False)
    persentase_probabilitas = db.Column(db.Float, nullable=False)
    urutan_ranking = db.Column(db.Integer, nullable=False)

    riwayat = db.relationship("RiwayatKlasifikasi", back_populates="probabilitas")
    kelas = db.relationship("KelasKhat")

    __table_args__ = (
        db.UniqueConstraint("id_riwayat", "id_kelas", name="uq_prob_pred_kelas"),
    )

    @property
    def classification_id(self):
        return self.id_riwayat

    @classification_id.setter
    def classification_id(self, value):
        self.id_riwayat = value

    @property
    def class_id(self):
        return self.id_kelas

    @class_id.setter
    def class_id(self, value):
        self.id_kelas = value

    @property
    def softmax_score(self):
        return self.skor_softmax

    @softmax_score.setter
    def softmax_score(self, value):
        self.skor_softmax = value

    @property
    def probability_percent(self):
        return self.persentase_probabilitas

    @probability_percent.setter
    def probability_percent(self, value):
        self.persentase_probabilitas = value

    @property
    def rank_order(self):
        return self.urutan_ranking

    @rank_order.setter
    def rank_order(self, value):
        self.urutan_ranking = value

    @property
    def khat_class(self):
        return self.kelas


PredictionProbability = ProbabilitasPrediksi
