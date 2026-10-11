import unittest

import numpy as np

import motorlib.grains
import motorlib.motor
import motorlib.propellant
from motorlib.burnmodels import ErosiveBurn, SteadyBurn
from motorlib.grain import ConvertedGrain
from motorlib.simResult import SimAlertLevel

_PROPELLANT = {
    'name': 'Test', 'density': 1700,
    'tabs': [{'minPressure': 0, 'maxPressure': 2e7, 'a': 1.569e-5, 'n': 0.4, 'k': 1.19, 't': 3105, 'm': 24.98}],
}
_EROSIVE_INPUTS = {'gasSpecificHeat': 2050.0, 'gasViscosity': 9.2e-5, 'gasConductivity': 0.37, 'solidSpecificHeat': 1700.0}
_MAP_DIM = 32
_TIMESTEP = 0.03


def _motor(core, length, diameter, throat, erosive, convert=False, grains=2, inputs=True, timestep=_TIMESTEP):
    motor = motorlib.motor.Motor()
    motor.config.setProperties({
        'maxPressure': 3e7, 'maxMassFlux': 1e4, 'maxMachNumber': 1, 'minPortThroat': 1, 'flowSeparationWarnPercent': 0.05,
        'burnoutWebThres': 2.5e-5, 'burnoutThrustThres': 1.0, 'timestep': timestep, 'ambPressure': 101325, 'mapDim': 750,
        '3DmapDim': _MAP_DIM, 'sepPressureRatio': 0.4, 'convertTo3D': convert, 'erosiveBurning': erosive,
    })
    motor.nozzle.setProperties({'throat': throat, 'exit': throat * 2.4, 'efficiency': 0.95, 'divAngle': 15, 'convAngle': 45,
                                'throatLength': 0.005, 'slagCoeff': 0, 'erosionCoeff': 0})
    motor.propellant = motorlib.propellant.Propellant({**_PROPELLANT, **(_EROSIVE_INPUTS if inputs else {})})
    for _ in range(grains):
        grain = motorlib.grains.grainTypes['BATES']()
        grain.setProperties({'diameter': diameter, 'length': length, 'coreDiameter': core, 'inhibitedEnds': 'Both'})
        motor.grains.append(grain)
    return motor


def _wideBore(**kwargs):
    return _motor(0.03, 0.12, 0.08, 0.012, **kwargs)


def _longBore(**kwargs):
    return _motor(0.020, 0.50, 0.05, 0.0183, grains=1, **kwargs)


class TestErosiveInputs(unittest.TestCase):
    def test_erosive_burning_is_off_by_default_and_old_files_load(self):
        self.assertFalse(motorlib.motor.MotorConfig().getProperty('erosiveBurning'))
        motor = motorlib.motor.Motor()
        motor.applyDict({'nozzle': {}, 'propellant': {'name': 'Old', 'density': 1700, 'tabs': _PROPELLANT['tabs']},
                         'grains': [], 'config': {'timestep': 0.05}})
        self.assertFalse(motor.config.getProperty('erosiveBurning'))
        self.assertEqual(len(motor.propellant.getErosiveErrors()), 1)

    def test_a_propellant_without_the_inputs_reports_one_error_naming_them(self):
        errors = motorlib.propellant.Propellant(_PROPELLANT).getErosiveErrors()
        self.assertEqual(len(errors), 1)
        self.assertIs(errors[0].level, SimAlertLevel.ERROR)
        for name in ('gas specific heat', 'gas viscosity', 'gas thermal conductivity', 'propellant specific heat'):
            self.assertIn(name, errors[0].description)

    def test_a_propellant_with_the_inputs_has_no_erosive_errors(self):
        self.assertEqual(motorlib.propellant.Propellant({**_PROPELLANT, **_EROSIVE_INPUTS}).getErosiveErrors(), [])

    def test_inputs_survive_a_save_and_load(self):
        motor = _wideBore(erosive=True)
        copy = motorlib.motor.Motor(motor.getDict())
        self.assertEqual(copy.propellant.getProperty('gasConductivity'), 0.37)
        self.assertTrue(copy.config.getProperty('erosiveBurning'))
        self.assertEqual(copy.config.getProperty('erosiveRoughness'), motor.config.getProperty('erosiveRoughness'))


class TestErosiveSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wideErosive = _wideBore(erosive=True).runSimulation()
        cls.widePlain = _wideBore(erosive=False, convert=True).runSimulation()
        cls.longErosive = _longBore(erosive=True).runSimulation()
        cls.longPlain = _longBore(erosive=False, convert=True).runSimulation()

    def test_a_wide_bore_burns_as_it_does_without_erosion(self):
        self.assertTrue(self.wideErosive.success)
        self.assertAlmostEqual(self.wideErosive.getMaxPressure() / self.widePlain.getMaxPressure(), 1, delta=0.05)
        self.assertAlmostEqual(self.wideErosive.getImpulse() / self.widePlain.getImpulse(), 1, delta=0.06)

    @staticmethod
    def _earlyPressure(result):
        """Pressure a few steps in, while the bore is at its narrowest and the gas flows fastest through it."""
        return result.channels['pressure'].getPoint(3)

    def test_a_long_narrow_bore_starts_at_a_higher_pressure_and_gives_more_impulse_than_without_erosion(self):
        self.assertGreater(self._earlyPressure(self.longErosive), 1.5 * self._earlyPressure(self.longPlain))
        self.assertGreater(self.longErosive.getImpulse(), 1.04 * self.longPlain.getImpulse())
        self.assertGreater(self.longErosive.getPeakMachNumber(), 1.2 * self.widePlain.getPeakMachNumber())

    def test_stronger_gas_heat_transfer_means_more_erosion(self):
        stronger = _longBore(erosive=True)
        stronger.propellant.setProperty('gasConductivity', 1.1)
        self.assertGreater(self._earlyPressure(stronger.runSimulation()), 1.1 * self._earlyPressure(self.longErosive))

    def test_the_result_has_the_normal_channels_on_the_motors_timestep(self):
        result = self.wideErosive
        times = np.array(result.channels['time'].getData())
        np.testing.assert_allclose(times, _TIMESTEP * np.arange(len(times)), atol=1e-9)
        for name, channel in result.channels.items():
            self.assertEqual(len(channel.getData()), len(times), name)
        for name in ('mass', 'massFlow', 'massFlux', 'regression', 'web', 'machNumber'):
            self.assertEqual({len(point) for point in result.channels[name].getData()}, {2}, name)

    def test_grains_lose_mass_and_web_and_gain_regression_as_the_motor_fires(self):
        channels = self.wideErosive.channels
        mass, web, regression = (np.array(channels[k].getData()) for k in ('mass', 'web', 'regression'))
        self.assertTrue(np.all(np.diff(mass, axis=0) <= 1e-9))
        self.assertTrue(np.all(np.diff(web, axis=0) <= 1e-9))
        self.assertTrue(np.all(np.diff(regression, axis=0) >= -1e-9))
        self.assertLess(float(mass[-1].sum()), 0.05 * float(mass[0].sum()))

    def test_the_flow_reports_the_mass_flowing_out_of_each_grain(self):
        massFlow = np.array(self.wideErosive.channels['massFlow'].getData())[1:6]
        self.assertTrue(np.all(massFlow[:, 0] > 0))
        self.assertTrue(np.all(massFlow[:, 1] > massFlow[:, 0]))

    def test_the_regular_checks_still_apply(self):
        motor = _wideBore(erosive=True)
        motor.config.setProperty('maxPressure', 1e6)
        descriptions = ' '.join(a.description for a in motor.runSimulation().alerts)
        self.assertIn('Max pressure exceeded', descriptions)

    def test_the_motor_and_its_grains_are_left_alone(self):
        motor = _wideBore(erosive=True)
        grains = list(motor.grains)
        result = motor.runSimulation()
        self.assertEqual(motor.grains, grains)
        self.assertTrue(all(isinstance(g, ConvertedGrain) for g in result.motor.grains))

    def test_a_shorter_timestep_gives_the_same_burn(self):
        fine = _wideBore(erosive=True, timestep=_TIMESTEP / 2).runSimulation()
        self.assertAlmostEqual(fine.getImpulse() / self.wideErosive.getImpulse(), 1, delta=0.03)
        self.assertAlmostEqual(fine.getMaxPressure() / self.wideErosive.getMaxPressure(), 1, delta=0.03)


class TestErosiveErrors(unittest.TestCase):
    def test_a_propellant_without_the_inputs_stops_the_simulation_with_an_error(self):
        result = _wideBore(erosive=True, inputs=False).runSimulation()
        self.assertFalse(result.success)
        errors = result.getAlertsByLevel(SimAlertLevel.ERROR)
        self.assertEqual(len(errors), 1)
        self.assertIn('Erosive burning needs', errors[0].description)

    def test_a_grain_without_a_3d_form_stops_the_simulation_with_an_error(self):
        motor = _wideBore(erosive=True, grains=1)
        endBurner = motorlib.grains.grainTypes['End Burner']()
        endBurner.setProperties({'diameter': 0.08, 'length': 0.05})
        motor.grains.insert(0, endBurner)
        result = motor.runSimulation()
        self.assertFalse(result.success)
        errors = result.getAlertsByLevel(SimAlertLevel.ERROR)
        self.assertEqual([e.location for e in errors], ['Grain 1'])
        self.assertIn('erosive burning', errors[0].description)

    def test_cancelling_from_the_progress_callback_ends_the_run_early(self):
        progress = []

        def callback(value):
            progress.append(value)
            return len(progress) >= 3

        result = _wideBore(erosive=True).runSimulation(callback)
        self.assertFalse(result.success)
        self.assertEqual(len(progress), 3)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in progress))
        self.assertEqual(progress, sorted(progress))

    def test_the_start_callback_is_called_once_before_the_first_step(self):
        calls = []
        result = _wideBore(erosive=True).runSimulation(onSimStart=lambda: calls.append(len(calls)))
        self.assertEqual(calls, [0])
        self.assertTrue(result.success)


class TestBurnModels(unittest.TestCase):
    def test_the_config_picks_the_model(self):
        self.assertIsInstance(_wideBore(erosive=False)._makeBurnModel(), SteadyBurn)
        motor = _wideBore(erosive=True).getWithGrainsConvertedTo3D()
        for grain in motor.grains:
            grain.simulationSetup(motor.config)
        self.assertIsInstance(motor._makeBurnModel(), ErosiveBurn)

    def test_the_steady_model_reports_a_pressure_and_per_grain_values_at_rest(self):
        motor = _wideBore(erosive=False)
        for grain in motor.grains:
            grain.simulationSetup(motor.config)
        step = SteadyBurn(motor).initial()
        self.assertGreater(step.pressure, 0)
        self.assertEqual(step.regression, [0, 0])
        self.assertEqual(len(step.mass), 2)
        self.assertIsNone(step.machNumber)
