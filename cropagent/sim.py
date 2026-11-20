"""Thin wrapper around PCSE's LINTUL3 (N-limited spring wheat) for step-wise control."""
import os, warnings, yaml
warnings.filterwarnings("ignore")
import pcse
from pcse import signals
from pcse.engine import Engine
from pcse.base import ParameterProvider
from pcse.input import PCSEFileReader, CABOWeatherDataProvider

# Crop/site/soil/weather files are copied from PCSE's bundled test data (EUPL-1.2), see data/lintul3/NOTICE.
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "lintul3")
# Years with complete Mar-Oct coverage (1990, 1991 have gaps in the bundled files).
VALID_YEARS = list(range(1976, 1990)) + list(range(1992, 2000))
STEP_DAYS = 7
MAX_STEPS = 30

_cache = {}
def _static():
    if not _cache:
        _cache["site"] = PCSEFileReader(os.path.join(DATA, "lintul3_springwheat.site"))
        _cache["soil"] = PCSEFileReader(os.path.join(DATA, "lintul3_springwheat.soil"))
        _cache["crop"] = PCSEFileReader(os.path.join(DATA, "lintul3_springwheat.crop"))
        _cache["wdp"] = CABOWeatherDataProvider("NL1", DATA, ETmodel="P")
    return _cache

def agro(year):
    return yaml.safe_load(f"""
Version: 1.0
AgroManagement:
- {year}-03-25:
    CropCalendar:
        crop_name: wheat
        variety_name: spring-wheat
        crop_start_date: {year}-03-31
        crop_start_type: emergence
        crop_end_date: {year}-10-20
        crop_end_type: earliest
        max_duration: 366
    TimedEvents: null
    StateEvents: null
""")["AgroManagement"]

def make_engine(year):
    c = _static()
    pv = ParameterProvider(sitedata=c["site"], soildata=c["soil"], cropdata=c["crop"])
    e = Engine(pv, c["wdp"], agromanagement=agro(year), config="Lintul3.conf")
    return e

def apply_n(engine, amount_g_m2, recovery=0.7):
    # NOTE: uses PCSE's private Engine._send_signal (works with the pinned pcse==6.0.13; may break on other versions).
    if amount_g_m2 > 0:
        engine._send_signal(signal=signals.apply_n, amount=float(amount_g_m2), recovery=recovery)

def run_plan(year, plan):
    """plan: {weekly_step_index: gN/m2}. Returns final storage-organ weight (g/m2)."""
    e = make_engine(year); e.run(days=1); k = 0
    while not e.flag_terminate and k < 60:
        apply_n(e, plan.get(k, 0.0)); e.run(days=STEP_DAYS); k += 1
    return e.get_output()[-1]["WSO"]
