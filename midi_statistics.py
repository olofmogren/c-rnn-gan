# Tools to load and save midi files for the rnn-gan-project.
# 
# Written by Olof Mogren, http://mogren.one/
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import sys, os, midi, math, string, time

GENRE      = 0
COMPOSER   = 1
SONG_DATA  = 2

# INDICES IN BATCHES:
LENGTH     = 0
FREQ       = 1
VELOCITY   = 2
TICKS_FROM_PREV_START      = 3

# INDICES IN SONG DATA (NOT YET BATCHED):
BEGIN_TICK = 3
CHANNEL    = 4

debug = ''
#debug = 'overfit'


base_tones = {'C':   0,
              'C#':  1, 
              'D':   2,
              'D#':  3,
              'E':   4,
              'F':   5,
              'F#':  6,
              'G':   7,
              'G#':  8,
              'A':   9,
              'A#': 10,
              'B':  11}

scale = {}
#Major scale:
scale['major'] = [0,2,4,5,7,9,11]
#(W-W-H-W-W-W-H)
#(2 2 1 2 2 2 1)

#Natural minor scale:
scale['natural_minor'] = [0,2,3,5,7,8,10]
#(W-H-W-W-H-W-W)
#(2 1 2 2 1 2 2)
 
#Harmonic minor scale:
scale['harmonic_minor'] = [0,2,3,5,7,8,11]
#(W-H-W-W-H-WH-H)
#(2 1 2 2 1 3 1)

tone_names = {}
for tone_name in base_tones:
  tone_names[base_tones[tone_name]] = tone_name


def get_tones(midi_pattern):
  """
  returns a dict of statistics, keys: [scale_distribution,
  """
  
  tones = []
  
  for track in midi_pattern:
    for event in track:
      if type(event) == midi.events.SetTempoEvent:
        pass # These are currently ignored
      elif (type(event) == midi.events.NoteOffEvent) or \
           (type(event) == midi.events.NoteOnEvent and \
            event.velocity == 0):
        pass # not needed here
      elif type(event) == midi.events.NoteOnEvent:
        tones.append(event.data[0])
  return tones 

def detect_beat(midi_pattern):
  """
  returns a dict of statistics, keys: [scale_distribution,
  """
  
  abs_ticks = []
  
  # Tempo:
  ticks_per_quarter_note = float(midi_pattern.resolution)
  
  for track in midi_pattern:
    abs_tick=0
    for event in track:
      abs_tick += event.tick
      if type(event) == midi.events.SetTempoEvent:
        pass # These are currently ignored
      elif (type(event) == midi.events.NoteOffEvent) or \
           (type(event) == midi.events.NoteOnEvent and \
            event.velocity == 0):
        pass
      elif type(event) == midi.events.NoteOnEvent:
        abs_ticks.append(abs_tick)
  stats = {}
  for quarter_note_estimate in range(int(ticks_per_quarter_note), int(0.75*ticks_per_quarter_note), -1):
    #print('est: {}'.format(quarter_note_estimate))
    avg_ticks_off = []
    for begin_tick in range(quarter_note_estimate):
      ticks_off = []
      for abs_tick in abs_ticks:
        #print('abs_tick: {} % {}'.format(abs_tick, quarter_note_estimate/4))
        sixteenth_note_estimate = quarter_note_estimate//4
        ticks_off_sixteenths = int((begin_tick+abs_tick)%sixteenth_note_estimate)
        if ticks_off_sixteenths > sixteenth_note_estimate//2:
          # off, but before beat 
          ticks_off_sixteenths = -(ticks_off_sixteenths-sixteenth_note_estimate)
        #print('ticks_off: {}'.format(ticks_off_sixteenths))
        ticks_off.append(ticks_off_sixteenths)
      avg_ticks_off.append(float(sum(ticks_off))/float(len(ticks_off)))
      #print('avg_ticks_off: {}. min: {}.'.format(avg_ticks_off, min(avg_ticks_off)))
    stats[quarter_note_estimate] = min(avg_ticks_off)
  return stats

def get_abs_ticks(midi_pattern):
  abs_ticks = []
  for track in midi_pattern:
    abs_tick=0
    for event in track:
      abs_tick += event.tick
      if type(event) == midi.events.SetTempoEvent:
        pass # These are currently ignored
      elif (type(event) == midi.events.NoteOffEvent) or \
           (type(event) == midi.events.NoteOnEvent and \
            event.velocity == 0):
        pass
      elif type(event) == midi.events.NoteOnEvent:
        abs_ticks.append(abs_tick)
  abs_ticks.sort()
  return abs_ticks

def get_top_k_intervals(midi_pattern, k):
  """
  returns a fraction of the noteon events in midi_pattern that are polyphonous
  (several notes occurring at the same time).
  Here, two note on events are counted as the same event if they
  occur at the same time, and in this case it is considered a polyphonous event.
  """
  intervals = {}
  abs_ticks = get_abs_ticks(midi_pattern)
  accumulator = 0
  last_abs_tick = 0
  for abs_tick in abs_ticks:
    interval = abs_tick-last_abs_tick
    if interval not in intervals:
      intervals[interval] = 0
    intervals[interval] += 1
    accumulator += 1
    last_abs_tick = abs_tick
  intervals_list = [(interval, intervals[interval]/float(accumulator)) for interval in intervals]
  intervals_list.sort(key=lambda i: i[1], reverse=True)
  return intervals_list[:k]


def get_polyphony_score(midi_pattern):
  """
  returns a fraction of the noteon events in midi_pattern that are polyphonous
  (several notes occurring at the same time).
  Here, two note on events are counted as the same event if they
  occur at the same time, and in this case it is considered a polyphonous event.
  """
  
  abs_ticks = get_abs_ticks(midi_pattern)
  monophonous_events = 0
  polyphonous_events = 0
  
  last_abs_tick = 0
  tones_in_current_event = 0
  for abs_tick in abs_ticks:
    if abs_tick == last_abs_tick:
      tones_in_current_event += 1
    else:
      if tones_in_current_event == 1:
        monophonous_events += 1
      elif tones_in_current_event > 1:
        polyphonous_events += 1
      tones_in_current_event = 1
    last_abs_tick = abs_tick
    if tones_in_current_event == 1:
      monophonous_events += 1
    elif tones_in_current_event > 1:
      polyphonous_events += 1
  if polyphonous_events == 0:
    return 0.0
  return float(polyphonous_events)/(polyphonous_events+monophonous_events)


def get_rhythm_stats(midi_pattern):
  """
  returns a dict of statistics, keys: [scale_distribution,
  """
  
  abs_ticks = []
  
  # Tempo:
  ticks_per_quarter_note = float(midi_pattern.resolution)
  
  # Multiply with output_ticks_pr_input_tick for output ticks.
  for track in midi_pattern:
    abs_tick=0
    for event in track:
      abs_tick += event.tick
      if type(event) == midi.events.SetTempoEvent:
        pass # These are currently ignored
      elif (type(event) == midi.events.NoteOffEvent) or \
           (type(event) == midi.events.NoteOnEvent and \
            event.velocity == 0):
        pass
      elif type(event) == midi.events.NoteOnEvent:
        abs_ticks.append(abs_tick)
  stats = {}
  for abs_tick in abs_ticks:
    ticks_since_quarter_note = int(abs_tick%ticks_per_quarter_note)
    if ticks_since_quarter_note not in stats:
      stats[ticks_since_quarter_note] = 1
    else:
      stats[ticks_since_quarter_note] += 1
  return stats


def get_intensities(midi_pattern):
  """
  returns a dict of statistics, keys: [scale_distribution,
  """
  
  intensities = []
  
  for track in midi_pattern:
    abs_tick=0
    for event in track:
      abs_tick += event.tick
      if type(event) == midi.events.SetTempoEvent:
        pass # These are currently ignored
      elif (type(event) == midi.events.NoteOffEvent) or \
           (type(event) == midi.events.NoteOnEvent and \
            event.velocity == 0):
        pass
      elif type(event) == midi.events.NoteOnEvent:
        intensities.append(event.velocity)
  return (min(intensities), max(intensities))


def get_midi_pattern(filename):
  try:
    return midi.read_midifile(filename)
  except:
    print ('Error reading {}'.format(filename))
    return None

def tones_to_scales(tones):
  """
   Midi to tone name (octave: -5):
   0: C
   1: C#
   2: D
   3: D#
   4: E
   5: F
   6: F#
   7: G
   8: G#
   9: A
   10: A#
   11: B

   Melodic minor scale is ignored.

   One octave is 12 tones.
  """
  counts = {}
  for base_tone in base_tones:
    counts[base_tone] = {}
    counts[base_tone]['major'] = 0
    counts[base_tone]['natural_minor'] = 0
    counts[base_tone]['harmonic_minor'] = 0

  if not len(tones):
    frequencies = {}
    for base_tone in base_tones:
      frequencies[base_tone] = {}
      for scale_label in scale:
        frequencies[base_tone][scale_label] = 0.0
    return frequencies
  for tone in tones:
    for base_tone in base_tones:
      for scale_label in scale:
        if tone%12-base_tones[base_tone] in scale[scale_label]:
          counts[base_tone][scale_label] += 1
  frequencies = {}
  for base_tone in counts:
    frequencies[base_tone] = {}
    for scale_label in counts[base_tone]:
      frequencies[base_tone][scale_label] = float(counts[base_tone][scale_label])/float(len(tones))
  return frequencies

def repetitions(tones):
  rs = {}
  #print(tones)
  #print(len(tones)/2)
  for l in range(2, min(len(tones)//2, 10)):
    #print (l)
    rs[l] = 0
    for i in range(len(tones)-l*2):
      for j in range(i+l,len(tones)-l):
        #print('comparing \'{}\' and \'{}\''.format(tones[i:i+l], tones[j:j+l]))
        if tones[i:i+l] == tones[j:j+l]:
          rs[l] += 1
  rs2 = {}
  for r in rs:
    if rs[r]:
      rs2[r] = rs[r]
  return rs2
      

def tone_to_tone_name(tone):
  """
   Midi to tone name (octave: -5):
   0: C
   1: C#
   2: D
   3: D#
   4: E
   5: F
   6: F#
   7: G
   8: G#
   9: A
   10: A#
   11: B

   One octave is 12 tones.
  """

  base_tone = tone_names[tone%12]
  octave = tone//12-5
  return '{} {}'.format(base_tone, octave)

def max_likelihood_scale(tones):
  scale_statistics = tones_to_scales(tones) 
  stat_list = []
  for base_tone in scale_statistics:
    for scale_label in scale_statistics[base_tone]:
      stat_list.append((base_tone, scale_label, scale_statistics[base_tone][scale_label]))
  stat_list.sort(key=lambda e: e[2], reverse=True)
  return (stat_list[0][0]+' '+stat_list[0][1], stat_list[0][2])

def tone_to_freq(tone):
  """
    returns the frequency of a tone. 

    formulas from
      * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
      * https://en.wikipedia.org/wiki/Cent_(music)
  """
  return math.pow(2, ((float(tone)-69.0)/12.0)) * 440.0

def freq_to_tone(freq):
  """
    returns a dict d where
    d['tone'] is the base tone in midi standard
    d['cents'] is the cents to make the tone into the exact-ish frequency provided.
               multiply this with 8192 to get the midi pitch level.

    formulas from
      * https://en.wikipedia.org/wiki/MIDI_Tuning_Standard
      * https://en.wikipedia.org/wiki/Cent_(music)
  """
  if freq == 0.0:
    return None
  float_tone = (69.0+12*math.log(float(freq)/440.0, 2))
  int_tone = int(float_tone)
  cents = int(1200*math.log(float(freq)/tone_to_freq(int_tone), 2))
  return {'tone': int_tone, 'cents': cents}

def cents_to_pitchwheel_units(cents):
  return int(40.96*(float(cents)))

def get_all_stats(midi_pattern):
  stats = {}
  if not midi_pattern:
    print('Failed to read midi pattern.')
    return None
  tones = get_tones(midi_pattern)
  if len(tones) == 0:
    print('This is an empty song.')
    return None
  stats['num_tones'] = len(tones)
  stats['tone_min'] = min(tones)
  stats['freq_min'] = tone_to_freq(min(tones))
  stats['tone_max'] = max(tones)
  stats['freq_max'] = tone_to_freq(max(tones))
  stats['tone_span'] = max(tones)-min(tones)
  stats['freq_span'] = tone_to_freq(max(tones))-tone_to_freq(min(tones))
  stats['tones_unique'] = len(set(tones))
  rs = repetitions(tones)
  for r in range(2,10):
    if r in rs:
      stats['repetitions_{}'.format(r)] = rs[r]
    else:
      stats['repetitions_{}'.format(r)] = 0
  
  ml = max_likelihood_scale(tones)
  stats['scale'] = ml[0]
  stats['scale_score'] = ml[1]
  
  beat_stats = detect_beat(midi_pattern)
  minval = float(midi_pattern.resolution)
  argmin = -1
  for beat in beat_stats:
    #print('Looking at beat: {}. Avg ticks off: {}.'.format(beat, beat_stats[beat]))
    if beat_stats[beat] < minval:
      minval = beat_stats[beat]
      argmin = beat
  stats['estimated_beat'] = argmin
  stats['estimated_beat_avg_ticks_off'] = minval
  (min_int, max_int) = get_intensities(midi_pattern)
  stats['intensity_min'] = min_int
  stats['intensity_max'] = max_int
  stats['intensity_span'] = max_int-min_int

  stats['polyphony_score'] = get_polyphony_score(midi_pattern)
  stats['top_10_intervals'] = get_top_k_intervals(midi_pattern, 10)
  stats['top_2_interval_difference'] = 0.0
  if len(stats['top_10_intervals']) > 1:
    stats['top_2_interval_difference'] = abs(stats['top_10_intervals'][1][0]-stats['top_10_intervals'][0][0])
  stats['top_3_interval_difference'] = 0.0
  if len(stats['top_10_intervals']) > 2:
    stats['top_3_interval_difference'] = abs(stats['top_10_intervals'][2][0]-stats['top_10_intervals'][0][0])

  return stats

def get_gnuplot_line(midi_patterns, i, showheader=True):
  stats = []
  print('#getting stats...')
  stats_time = time.time()
  for p in midi_patterns:
    stats.append(get_all_stats(p))
  print('done. time: {}'.format(time.time()-stats_time))
  #print(stats)
  stats_keys_string = ['scale']
  stats_keys = ['scale_score', 'tone_min', 'tone_max', 'tone_span', 'freq_min', 'freq_max', 'freq_span', 'tones_unique', 'repetitions_2', 'repetitions_3', 'repetitions_4', 'repetitions_5', 'repetitions_6', 'repetitions_7', 'repetitions_8', 'repetitions_9', 'estimated_beat', 'estimated_beat_avg_ticks_off', 'intensity_min', 'intensity_max', 'intensity_span', 'polyphony_score', 'top_2_interval_difference', 'top_3_interval_difference', 'num_tones']
  gnuplotline = ''
  if showheader:
    gnuplotline = '# global-step {} {}\n'.format(' '.join([s.replace(' ', '_') for s in stats_keys_string]), ' '.join(stats_keys))
  gnuplotline += '{} {} {}\n'.format(i, ' '.join(['{}'.format(stats[0][key].replace(' ', '_')) for key in stats_keys_string]), ' '.join(['{:.3f}'.format(sum([s[key] for s in stats])/float(len(stats))) for key in stats_keys]))
  return gnuplotline



def main():
  if len(sys.argv) > 2 and sys.argv[1] == '--gnuplot':
    #number = sys.argv[2]
    patterns = []
    for i in range(3,len(sys.argv)):
      #print(i)
      filename = sys.argv[i]
      print('#File: {}'.format(filename))
      #patterns.append(get_midi_pattern(filename))
      print(get_gnuplot_line([get_midi_pattern(filename)], i, showheader=(i==0)))
    
  else:
    for i in range(1,len(sys.argv)):
      filename = sys.argv[i]
      print('File: {}'.format(filename))
      midi_pattern = get_midi_pattern(filename)
      stats = get_all_stats(midi_pattern)
      if stats is None:
        print('Could not extract stats.')
      else:
        print ('ML scale estimate: {}: {:.2f}'.format(stats['scale'], stats['scale_score']))
        print ('Min tone: {}, {:.1f} Hz.'.format(tone_to_tone_name(stats['tone_min']), stats['freq_min']))
        print ('Max tone: {}, {:.1f} Hz.'.format(tone_to_tone_name(stats['tone_max']), stats['freq_max']))
        print ('Span: {} tones, {:.1f} Hz.'.format(stats['tone_span'], stats['freq_span']))
        print ('Unique tones: {}'.format(stats['tones_unique']))
        for r in xrange(2,10):
          print('Repetitions of len {}: {}'.format(r, stats['repetitions_{}'.format(r)]))
        print('Estimated beat: {}. Avg ticks off: {:.2f}.'.format(stats['estimated_beat'], stats['estimated_beat_avg_ticks_off']))
        print('Intensity: min: {}, max: {}.'.format(stats['intensity_min'], stats['intensity_max']))
        print('Polyphonous events: {:.2f}.'.format(stats['polyphony_score']))
        print('Top intervals:')
        for interval,score in stats['top_10_intervals']:
          print('{}: {:.2f}.'.format(interval,score))
        print('Top 2 interval difference: {}.'.format(stats['top_2_interval_difference']))
        print('Top 3 interval difference: {}.'.format(stats['top_3_interval_difference']))


if __name__ == "__main__":
  main()

