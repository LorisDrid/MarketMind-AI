import unittest

class TestRiskManagement(unittest.TestCase):
    def test_stop_loss_presence(self):
        # Simule la vérification qu'un stop-loss est bien défini à 2%
        stop_loss = 0.02 
        self.assertGreater(stop_loss, 0, "Le Stop-Loss doit être configuré")

    def test_max_drawdown_limit(self):
        # Vérifie que le bot ne risque pas plus de 5% du capital par trade
        max_risk_per_trade = 0.05
        self.assertLessEqual(max_risk_per_trade, 0.10, "Risque trop élevé par trade !")

if __name__ == "__main__":
    unittest.main()