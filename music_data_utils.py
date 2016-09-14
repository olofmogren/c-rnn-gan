import urllib2, re, os

# The following frequencies was fetched from http://www.deimos.ca/notefreqs/
# List indices corresponds to midi tone numbers, 0-127.
midi_frequencies = [8.1757989156,8.661957218,9.1770239974,9.7227182413,10.3008611535,10.9133822323,11.5623257097,12.2498573744,12.9782717994,13.75,14.5676175474,15.4338531643,16.3515978313,17.3239144361,18.3540479948,19.4454364826,20.6017223071,21.8267644646,23.1246514195,24.4997147489,25.9565435987,27.5,29.1352350949,30.8677063285,32.7031956626,34.6478288721,36.7080959897,38.8908729653,41.2034446141,43.6535289291,46.249302839,48.9994294977,51.9130871975,55,58.2704701898,61.735412657,65.4063913251,69.2956577442,73.4161919794,77.7817459305,82.4068892282,87.3070578583,92.4986056779,97.9988589954,103.826174395,110,116.5409403795,123.470825314,130.8127826503,138.5913154884,146.8323839587,155.563491861,164.8137784564,174.6141157165,184.9972113558,195.9977179909,207.65234879,220,233.081880759,246.9416506281,261.6255653006,277.1826309769,293.6647679174,311.1269837221,329.6275569129,349.228231433,369.9944227116,391.9954359817,415.3046975799,440,466.1637615181,493.8833012561,523.2511306012,554.3652619537,587.3295358348,622.2539674442,659.2551138257,698.456462866,739.9888454233,783.9908719635,830.6093951599,880,932.3275230362,987.7666025122,1046.5022612024,1108.7305239075,1174.6590716696,1244.5079348883,1318.5102276515,1396.912925732,1479.9776908465,1567.981743927,1661.2187903198,1760,1864.6550460724,1975.5332050245,2093.0045224048,2217.461047815,2349.3181433393,2489.0158697767,2637.020455303,2793.825851464,2959.9553816931,3135.963487854,3322.4375806396,3520,3729.3100921447,3951.066410049,4186.0090448096,4434.92209563,4698.6362866785,4978.0317395533,5274.0409106059,5587.6517029281,5919.9107633862,6271.926975708,6644.8751612791,7040,7458.6201842895,7902.132820098,8372.0180896192,8869.8441912599,9397.2725733571,9956.0634791066,10548.0818212119,11175.3034058561,11839.8215267723,12543.853951416]

sources                              = {}
sources['classical']                 = {}
sources['classical']['misc']         = ['http://www.midiworld.com/classic.htm']
sources['classical']['alkan']        = ['http://www.kunstderfuge.com/alkan.htm']
sources['classical']['bach']         = ['http://www.midiworld.com/bach.htm','http://www.kunstderfuge.com/bach/harpsi.htm','http://www.kunstderfuge.com/bach/wtk2.htm','http://www.kunstderfuge.com/bach/wtk1.htm','http://www.kunstderfuge.com/bach/organ.htm','http://www.kunstderfuge.com/bach/chamber.htm','http://www.kunstderfuge.com/bach/canons.htm','http://www.kunstderfuge.com/bach/chorales.htm','http://www.kunstderfuge.com/bach/variae.htm']
sources['classical']['bartok']       = ['http://www.midiworld.com/bartok.htm']
sources['classical']['beethoven']    = ['http://www.midiworld.com/beethoven.htm','http://www.kunstderfuge.com/beethoven/klavier.htm','http://www.kunstderfuge.com/beethoven/chamber.htm','http://www.kunstderfuge.com/beethoven/variae.htm']
sources['classical']['brahms']       = ['http://www.midiworld.com/brahms.htm','http://www.kunstderfuge.com/brahms.htm']
sources['classical']['byrd']         = ['http://www.midiworld.com/byrd.htm','http://www.kunstderfuge.com/byrd.htm']
sources['classical']['chopin']       = ['http://www.midiworld.com/chopin.htm','http://www.kunstderfuge.com/chopin.htm']
sources['classical']['debussy']      = ['http://www.kunstderfuge.com/debussy.htm']
sources['classical']['haydn']        = ['http://www.midiworld.com/haydn.htm','http://www.kunstderfuge.com/haydn.htm']
sources['classical']['handel']       = ['http://www.midiworld.com/handel.htm','http://www.kunstderfuge.com/handel.htm']
sources['classical']['hummel']       = ['http://www.midiworld.com/hummel.htm']
sources['classical']['liszt']        = ['http://www.midiworld.com/liszt.htm','http://www.kunstderfuge.com/liszt.htm']
sources['classical']['mendelssohn']  = ['http://www.midiworld.com/mendelssohn.htm','http://www.kunstderfuge.com/mendelssohn.htm']
sources['classical']['mozart']       = ['http://www.midiworld.com/mozart.htm','http://www.kunstderfuge.com/mozart.htm']
sources['classical']['rachmaninov']  = ['http://www.midiworld.com/rachmaninov.htm']
sources['classical']['raff']         = ['http://www.kunstderfuge.com/raff.htm']
sources['classical']['ravel']        = ['http://www.kunstderfuge.com/ravel.htm']
sources['classical']['satie']        = ['http://www.kunstderfuge.com/satie.htm']
sources['classical']['scarlatti']    = ['http://www.midiworld.com/scarlatti.htm']
sources['classical']['schubert']     = ['http://www.kunstderfuge.com/schubert.htm']
sources['classical']['schumann']     = ['http://www.midiworld.com/schumann.htm','http://www.kunstderfuge.com/schumann.htm']
sources['classical']['scriabin']     = ['http://www.midiworld.com/scriabin.htm']
sources['classical']['shostakovich'] = ['http://www.midiworld.com/shostakovich.htm']
sources['classical']['tchaikovsky']  = ['http://www.midiworld.com/tchaikovsky.htm','http://www.kunstderfuge.com/tchaikovsky.htm']
sources['classical']['earlymusic']   = ['http://www.midiworld.com/earlymus.htm']

def download_midi_data(datadir):
  midi_files = {}

  for genre in sources:
    midi_files[genre] = {}
    for composer in sources[genre]:
      midi_files[genre][composer] = []
      for url in sources[genre][composer]:
        response = urllib2.urlopen(url)
        #headers = response.info()
        data = response.read()
        links = re.findall('"(http://www.midiworld.com/midis/[^"]+\.mid)"', data)
        #print links
        for link in links:
          print link
          filename = link.split('/')[-1]
          print filename
          midi_files[genre][composer].append(filename)
          localdir = os.path.join(os.path.join(datadir, genre), composer)
          localpath = os.path.join(localdir, filename)
          if os.path.exists(localpath):
            print 'File exists. Not redownloading: {}'.format(localpath)
          else:
            response_midi = urllib2.urlopen(link)
            try: os.makedirs(localdir)
            except: pass
            data_midi = response_midi.read()
            if 'DOCTYPE html PUBLIC' not in data_midi:
              with open(localpath, 'w') as f:
                f.write(data_midi)
            else:
              print 'Seems to have been served an html page instead of a midi file. Continuing with next file.'

def read_data(datadir):
  print 'Not yet implemented...'
  pass
