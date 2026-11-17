"""Pure unit tests of development-only transport-delay modeling.

No Unity, ROS graph, official bridge module or game reset is needed.
"""
import ast
from collections import deque
import math
import os
from pathlib import Path
import unittest

BRIDGE = (Path(__file__).resolve().parents[2] / "sdu_apex_autodrive" /
          "sdu_apex_autodrive" / "bridge_40hz.py")


def selected_namespace():
    tree = ast.parse(BRIDGE.read_text())
    names = {"DevCommandDelayBuffer", "_parse_dev_delay_ms"}
    chosen = ast.Module(body=[n for n in tree.body
                              if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                              and n.name in names], type_ignores=[])
    scope = {"deque": deque, "math": math, "os": os}
    exec(compile(chosen, str(BRIDGE), "exec"), scope)
    return scope


class DevLatencyModelTests(unittest.TestCase):
    def test_timestamps_and_zero_order_hold(self):
        delay = selected_namespace()["DevCommandDelayBuffer"](40)
        delay.append(100_000_000, "0.12", "0.5")
        self.assertIsNone(delay.mature(139_999_999))
        self.assertEqual(delay.mature(140_000_000),
                         (100_000_000, "0.12", "0.5"))
        self.assertIsNone(delay.mature(150_000_000))
        delay.append(160_000_000, "-0.42", "0.1")
        delay.append(170_000_000, "-0.43", "0.08")
        self.assertIsNone(delay.mature(190_000_000))
        self.assertEqual(delay.mature(230_000_000),
                         (170_000_000, "-0.43", "0.08"))

    def test_40hz_steps_apply_only_matured_command(self):
        buffer = selected_namespace()["DevCommandDelayBuffer"](30)
        outputs=[]
        applied="0.0"
        for step in range(6):
            t=step*25_000_000
            buffer.append(t,str(step/10),"0.3")
            item=buffer.mature(t)
            if item is not None:applied=item[1]
            outputs.append(applied)
        self.assertEqual(outputs,["0.0","0.0","0.0","0.1","0.2","0.3"])

    def test_reset_clears_unapplied_commands(self):
        buffer=selected_namespace()["DevCommandDelayBuffer"](25)
        buffer.append(0,"1.0","1.0")
        buffer.clear()
        self.assertIsNone(buffer.mature(100_000_000))
        with self.assertRaises(ValueError):
            buffer=selected_namespace()["DevCommandDelayBuffer"](25)
            buffer.append(10,"0","0")
            buffer.append(5,"1","1")

    def test_strict_parsing_and_zero_default(self):
        fn=selected_namespace()["_parse_dev_delay_ms"]
        key="SDU_APEX_DEV_SENSOR_DELAY_MS"
        previous=os.environ.pop(key,None)
        try:
            self.assertEqual(fn(key),0)
            for bad in ("abc","NaN","inf","-1","100000"):
                os.environ[key]=bad
                with self.assertRaises(ValueError):fn(key)
            os.environ[key]="26.5"
            self.assertEqual(fn(key),26.5)
        finally:
            if previous is None:os.environ.pop(key,None)
            else:os.environ[key]=previous

    def test_input_stamp_precedes_added_sensor_delay(self):
        source=BRIDGE.read_text()
        handler=source[source.index('def _packet_handler_body('):source.index('handlers["Bridge"] = packet_handler')]
        self.assertLess(handler.index('_capture_competition_packet(data)'),
                        handler.index('gevent.sleep(_dev_sensor_delay_s)'))
        self.assertLess(handler.index('gevent.sleep(_dev_sensor_delay_s)'),
                        handler.index('original_handler(sid, data)'))
        sender=source[source.index('def _run_command_sender('):source.index('def _install_packet_contract(')]
        self.assertLess(sender.index('_dev_command_delay_buffer.mature('),
                        sender.index('command = dict(_latest_command)'))
        self.assertIn('next_send += period',sender)


if __name__ == '__main__':unittest.main()
