Reusable pulse programs
=======================

A pulse program describes events and loops independently of the values
used for each simulation. Read it once with ``pulsar.read_sequence()`` and
call ``program.build(parameters)`` whenever the parameters change. The
result is a tuple of the existing ``Pulse``, ``Delay`` and ``Coherence``
objects, usable by the single-spin and coupled-spin simulators.

Parameters and expressions
---------------------------

Download :download:`hahn.seq <../../examples/simulation/hahn.seq>`
or save this Hahn echo as ``hahn.seq``::

    set rf=rf1
    repeat l1
        pulse p1 phase=ph1
        coherence -1
        delay d1
        pulse p2 phase=ph2
        delay d2
    end

Bind its independent pulse widths and delays in Python::

    from spectrochempy.nmr import pulsar

    program = pulsar.read_sequence("hahn.seq")
    parameters = {
        "p1": "8 us", "p2": "17 us",
        "d1": "87.5 us", "d2": "91.5 us",
        "rf1": "10 kHz", "ph1": 0, "ph2": 90, "l1": 1,
    }
    events = program.build(parameters)
    spectrum = pulsar.simulate_pulse_sequence(events, **settings)

``settings`` contains the sample and acquisition settings, such as spin,
Larmor frequency, interactions, spinning frequency and spectral width.
Durations and frequencies require explicit units: strings or quantities
from SpectroChemPy's unit registry. Bare numbers mean degrees for phases,
and dimensionless values for coherence orders and repeat counts. Ordinary
numeric scale factors such as ``8 * 1e-6`` do not carry units and are
rejected as durations; use ``"8 us"`` or ``8 * ur.us`` instead.

Optional ``set`` expressions may use previously bound names, real numbers,
``+``, ``-``, ``*``, ``/`` and parentheses. For instance::

    set rf=rf1
    set tau=n/vr
    set d1=tau-p1/2-p2/2
    set d2=tau-p2/2

These assignments are evaluated in order. Pulse durations remain independent
unless the program explicitly relates them. Unit names are ``s``, ``ms``,
``us``, ``ns``, ``Hz``, ``kHz``, ``MHz``, ``deg`` and ``rad``; both
``10 kHz`` and ``10*kHz`` are accepted. The syntax supports scalar arithmetic,
not Python function calls or arbitrary Python execution.

Loops and changing RF settings
-------------------------------

``repeat count`` blocks terminate with ``end`` and can be nested. A count
must evaluate to a nonnegative integer; zero skips the body. Assignments
inside a loop persist into later iterations::

    set phase=0
    repeat l1
        repeat l2
            pulse p1
            set phase=phase+90
        end
        delay d1
    end

Each ``build`` starts from a fresh copy of the supplied parameters. It never
changes the caller's mapping. The default expansion limit is 100,000 steps,
counting instructions and loop iterations; adjust ``max_steps`` explicitly
when needed. At most 16 nested loops are accepted.

RF, phase and offset settings can change anywhere in the sequence. For
example, an MQMAS preparation with a z filter can use three RF powers::

    set rf=rf_exc
    pulse p1
    coherence q
    delay d0
    set rf=rf_conv
    pulse p2 phase=ph2
    coherence 0
    delay dz
    set rf=rf_read
    pulse p3
    coherence 1

``set rf=...`` changes the current setting until another assignment. An
option such as ``pulse p2 rf=rf_conv`` overrides only that pulse. The same
rules apply to ``phase`` and ``offset``; their defaults are zero. At least
one of ``rf`` or ``partner_rf`` must be supplied for a pulse.

Simultaneous pulses on two nuclei
----------------------------------

For ``simulate_coupled_sequence``, one pulse can drive both RF-active nuclei
(:download:`two-channel example <../../examples/pulse_programs/two_channel.seq>`)::

    set rf=rf_observed
    set partner_rf=rf_partner
    pulse p1 phase=ph1 partner_phase=ph2 offset=o1 partner_offset=o2
    set rf=0 Hz
    pulse p2 partner_phase=ph3

The first pulse applies both fields simultaneously for the common duration
``p1``. The second drives only the partner for ``p2``. RF powers, phases and
offsets are independent for each channel. To describe unequal pulse lengths,
write the successive intervals of overlap explicitly. The reader does not
replace a simultaneous pulse by two sequential pulses. Choose the observed
and partner ``SpinSite`` objects in the Python simulation call.

Indirect dimensions and phase cycles
-------------------------------------

Place ``delay d0`` at the desired location in the pulse program and supply
its value for every indirect increment. This is actual free evolution of
the density matrix during preparation, followed by the remaining pulses.
It is independent of a repeated block inside one preparation.

For example, given a 3QMAS program that selects ``coherence q``, evolves
for ``d0`` and converts to detected order +1, the two complex States
branches can be calculated as follows::

    import numpy as np
    import spectrochempy as scp

    ur = scp.ur

    branches = []
    times = np.arange(48) * 10e-6
    for coefficients in ((0.5, 0.5), (-0.5j, 0.5j)):
        rows = []
        for t1 in times:
            scans = tuple(
                program.build({**parameters, "q": q, "d0": t1 * ur.s})
                for q in (3, -3)
            )
            cycle = pulsar.PhaseCycle(scans, coefficients)
            spectrum = pulsar.simulate_pulse_sequence(cycle, **settings)
            rows.append(spectrum.data)
        branches.append(np.stack(rows))

A :download:`z-filter MQMAS program <../../examples/pulse_programs/mqmas_zfilter.seq>`
also illustrates multiple RF settings and an indirect delay.

Each branch has shape ``(48, points)`` with complex values and represents
t1 evolution and the detected F2 spectrum. Keep both branches for subsequent
States processing along F1; plotting the real part is only a display choice.
The reader does not perform Fourier transformation, shearing or 2D processing.
For a preparation/variable-delay/conversion sequence, the existing
``simulate_t1_series`` API remains the optimized path; this general program
builder expands events and does not automatically factor that calculation.

Diagnostics
-------------

Syntax is checked when reading; parameters, units and calculated event
values are checked during ``build``. For example, a negative delay reports::

    hahn.seq:6: pulse duration must be positive; delay duration must be nonnegative

``SequenceError`` is a ``ValueError`` with ``source`` and one-based ``line``
attributes. Missing symbols, incompatible units, invalid loop counts and
division by zero also identify the relevant source line. ``#`` starts a
comment; indentation is cosmetic. Assignments in a skipped loop are not
executed. The file extension is unrestricted; this is a new pulse-program
format, not an interpreter for historical PyPulsar or TopSpin programs.
