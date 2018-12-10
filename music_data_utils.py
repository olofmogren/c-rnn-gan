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

try:
  from urllib.parse import urlparse
except ImportError:
  from urlparse import urlparse
from urllib.request import urlopen
import os, midi, math, random, re, string, sys
import numpy as np
from io import BytesIO

GENRE      = 0
COMPOSER   = 1
SONG_DATA  = 2

# INDICES IN BATCHES (LENGTH,FREQ,VELOCITY are repeated self.tones_per_cell times):
TICKS_FROM_PREV_START      = 0
LENGTH     = 1
FREQ       = 2
VELOCITY   = 3

# INDICES IN SONG DATA (NOT YET BATCHED):
BEGIN_TICK = 0

NUM_FEATURES_PER_TONE = 3

debug = ''
#debug = 'overfit'

sources                              = {}
sources['classical']                 = {}
sources['classical']['alkan']        = ['http://www.classicalmidi.co.uk/alkan.htm']
sources['classical']['adam']         = ['http://www.classicalmidi.co.uk/adam.htm']
sources['classical']['aguado']       = ['http://www.classicalmidi.co.uk/aguadodion.htm']
sources['classical']['albenizisaac'] = ['http://www.classicalmidi.co.uk/albeniz.htm']
sources['classical']['albenizmateo'] = ['http://www.classicalmidi.co.uk/albenizmateo.htm']
sources['classical']['albinoni']     = ['http://www.classicalmidi.co.uk/albinoni.htm']
sources['classical']['alford']       = ['http://www.classicalmidi.co.uk/alford.htm']
sources['classical']['anderson']     = ['http://www.classicalmidi.co.uk/anderson.htm']
sources['classical']['ansell']       = ['http://www.classicalmidi.co.uk/anselljohn.htm']
sources['classical']['arensky']      = ['http://www.classicalmidi.co.uk/arensky.htm']
sources['classical']['arriaga']      = ['http://www.classicalmidi.co.uk/arriag.htm']
sources['classical']['bach']         = ['http://www.midiworld.com/bach.htm','http://www.classicalmidi.co.uk/bach.htm']
sources['classical']['bartok']       = ['http://www.midiworld.com/bartok.htm','http://www.classicalmidi.co.uk/bartok.htm']
sources['classical']['barber']       = ['http://www.classicalmidi.co.uk/barber.htm']
sources['classical']['barbieri']     = ['http://www.classicalmidi.co.uk/barbie.htm']
sources['classical']['bax']          = ['http://www.classicalmidi.co.uk/bax.htm']
sources['classical']['beethoven']    = ['http://www.midiworld.com/beethoven.htm','http://www.classicalmidi.co.uk/beethoven.htm']
sources['classical']['bellini']      = ['http://www.classicalmidi.co.uk/bellini.htm']
sources['classical']['berlin']       = ['http://www.classicalmidi.co.uk/berlin.htm']
sources['classical']['berlioz']      = ['http://www.classicalmidi.co.uk/berlioz.htm']
sources['classical']['binge']        = ['http://www.classicalmidi.co.uk/binge.htm']
sources['classical']['bizet']        = ['http://www.classicalmidi.co.uk/bizet.htm']
sources['classical']['boccherini']   = ['http://www.classicalmidi.co.uk/bocc.htm']
sources['classical']['boellman']     = ['http://www.classicalmidi.co.uk/boell.htm']
sources['classical']['borodin']      = ['http://www.classicalmidi.co.uk/borodin.htm']
sources['classical']['boyce']        = ['http://www.classicalmidi.co.uk/boyce.htm']
sources['classical']['brahms']       = ['http://www.midiworld.com/brahms.htm','http://www.classicalmidi.co.uk/brahms.htm']
sources['classical']['breton']       = ['http://www.classicalmidi.co.uk/breton.htm']
sources['classical']['britten']      = ['http://www.classicalmidi.co.uk/britten.htm']
sources['classical']['bouwer']       = ['http://www.classicalmidi.co.uk/bouwer.htm']
sources['classical']['bruch']        = ['http://www.classicalmidi.co.uk/bruch.htm']
sources['classical']['bruckner']     = ['http://www.classicalmidi.co.uk/bruck.htm']
sources['classical']['bergmuller']   = ['http://www.classicalmidi.co.uk/bergmuller.htm']
sources['classical']['busoni']       = ['http://www.classicalmidi.co.uk/busoni.htm']
sources['classical']['byrd']         = ['http://www.midiworld.com/byrd.htm','http://www.classicalmidi.co.uk/byrd.htm']
sources['classical']['carulli']      = ['http://www.classicalmidi.co.uk/carull.htm']
sources['classical']['chabrier']     = ['http://www.classicalmidi.co.uk/chabrier.htm']
sources['classical']['chaminade']    = ['http://www.classicalmidi.co.uk/chaminad.htm']
sources['classical']['chapi']        = ['http://www.classicalmidi.co.uk/chapie.htm']
sources['classical']['cherubini']    = ['http://www.classicalmidi.co.uk/cherub.htm']
sources['classical']['chopin']       = ['http://www.midiworld.com/chopin.htm','http://www.classicalmidi.co.uk/chopin.htm']
sources['classical']['clementi']     = ['http://www.classicalmidi.co.uk/clemen.htm']
sources['classical']['coates']       = ['http://www.classicalmidi.co.uk/coates.htm']
sources['classical']['copland']      = ['http://www.classicalmidi.co.uk/copland.htm']
sources['classical']['corelli']      = ['http://www.classicalmidi.co.uk/cor.htm']
sources['classical']['cramer']       = ['http://www.classicalmidi.co.uk/cramer.htm']
sources['classical']['curzon']       = ['http://www.classicalmidi.co.uk/cuzon.htm']
sources['classical']['czerny']       = ['http://www.classicalmidi.co.uk/czerny.htm']
sources['classical']['debussy']      = ['http://www.classicalmidi.co.uk/debussy.htm']
sources['classical']['delibes']      = ['http://www.classicalmidi.co.uk/del.htm']
sources['classical']['delius']       = ['http://www.classicalmidi.co.uk/delius.htm']
sources['classical']['dialoc']       = ['http://www.classicalmidi.co.uk/diaoc.htm']
sources['classical']['dupre']        = ['http://www.classicalmidi.co.uk/dupre.htm']
sources['classical']['dussek']       = ['http://www.classicalmidi.co.uk/dussek.htm']
sources['classical']['dvorak']       = ['http://www.classicalmidi.co.uk/dvok.htm']
sources['classical']['elgar']        = ['http://www.classicalmidi.co.uk/elgar.htm']
sources['classical']['eshpai']       = ['http://www.classicalmidi.co.uk/Eshpai.htm', 'http://www.classicalmidi.co.uk/Eshpai%20.htm']
sources['classical']['faure']        = ['http://www.classicalmidi.co.uk/faure.htm']
sources['classical']['field']        = ['http://www.classicalmidi.co.uk/field.htm']
sources['classical']['flotow']       = ['http://www.classicalmidi.co.uk/flotow.htm']
sources['classical']['foster']       = ['http://www.classicalmidi.co.uk/foster.htm']
sources['classical']['franck']       = ['http://www.classicalmidi.co.uk/franck.htm']
sources['classical']['fresc']        = ['http://www.classicalmidi.co.uk/fresc.htm']
sources['classical']['garoto']       = ['http://www.classicalmidi.co.uk/garoto.htm']
sources['classical']['german']       = ['http://www.classicalmidi.co.uk/german.htm']
sources['classical']['gershwin']     = ['http://www.classicalmidi.co.uk/gershwin.htm']
sources['classical']['gilbert']      = ['http://www.classicalmidi.co.uk/gilbert.htm']
sources['classical']['ginast']       = ['http://www.classicalmidi.co.uk/ginast.htm']
sources['classical']['gott']         = ['http://www.classicalmidi.co.uk/gott.htm']
sources['classical']['gounod']       = ['http://www.classicalmidi.co.uk/gounod.htm']
sources['classical']['grain']        = ['http://www.classicalmidi.co.uk/grain.htm']
sources['classical']['grieg']        = ['http://www.classicalmidi.co.uk/grieg.htm']
sources['classical']['griff']        = ['http://www.classicalmidi.co.uk/griff.htm']
sources['classical']['haydn']        = ['http://www.midiworld.com/haydn.htm','http://www.classicalmidi.co.uk/haydn.htm']
sources['classical']['handel']       = ['http://www.midiworld.com/handel.htm','http://www.classicalmidi.co.uk/handel.htm']
sources['classical']['heller']       = ['http://www.classicalmidi.co.uk/heller.htm']
sources['classical']['herold']       = ['http://www.classicalmidi.co.uk/herold.htm']
sources['classical']['hiller']       = ['http://www.classicalmidi.co.uk/hiller.htm']
sources['classical']['holst']        = ['http://www.classicalmidi.co.uk/holst.htm']
sources['classical']['hummel']       = ['http://www.midiworld.com/hummel.htm','http://www.classicalmidi.co.uk/hummel.htm']
sources['classical']['ibert']        = ['http://www.classicalmidi.co.uk/ibert.htm']
sources['classical']['ives']         = ['http://www.classicalmidi.co.uk/ives.htm']
sources['classical']['janacek']      = ['http://www.classicalmidi.co.uk/janacek.htm']
sources['classical']['joplin']       = ['http://www.classicalmidi.co.uk/joplin.htm']
sources['classical']['jstrauss']     = ['http://www.classicalmidi.co.uk/jstrauss.htm']
sources['classical']['karg']         = ['http://www.classicalmidi.co.uk/karl.htm']
sources['classical']['khach']        = ['http://www.classicalmidi.co.uk/khach.htm']
sources['classical']['kuhlau']       = ['http://www.classicalmidi.co.uk/kuhlau.htm']
sources['classical']['lalo']         = ['http://www.classicalmidi.co.uk/lalo.htm']
sources['classical']['lemire']       = ['http://www.classicalmidi.co.uk/lemire.htm']
sources['classical']['lenar']        = ['http://www.classicalmidi.co.uk/lenar.htm']
sources['classical']['liszt']        = ['http://www.midiworld.com/liszt.htm','http://www.classicalmidi.co.uk/liszt.htm']
sources['classical']['lobos']        = ['http://www.classicalmidi.co.uk/lobos.htm']
sources['classical']['lovland']      = ['http://www.classicalmidi.co.uk/lovland.htm']
sources['classical']['lyssen']       = ['http://www.classicalmidi.co.uk/lyssen.htm']
sources['classical']['maccunn']      = ['http://www.classicalmidi.co.uk/maccunn.htm']
sources['classical']['mahler']       = ['http://www.classicalmidi.co.uk/mahler.htm']
sources['classical']['maier']        = ['http://www.classicalmidi.co.uk/maier.htm']
sources['classical']['marcello']     = ['http://www.classicalmidi.co.uk/marcello.htm']
sources['classical']['martini']      = ['http://www.classicalmidi.co.uk/martini.htm']
sources['classical']['mehul']        = ['http://www.classicalmidi.co.uk/mehul.htm']
sources['classical']['mendelssohn']  = ['http://www.midiworld.com/mendelssohn.htm','http://www.classicalmidi.co.uk/mend.htm']
sources['classical']['messager']     = ['http://www.classicalmidi.co.uk/messager.htm']
sources['classical']['messia']       = ['http://www.classicalmidi.co.uk/messia.htm']
sources['classical']['meyerbeer']    = ['http://www.classicalmidi.co.uk/meyerbeer.htm']
sources['classical']['modest']       = ['http://www.classicalmidi.co.uk/modest.htm']
sources['classical']['moszkowski']   = ['http://www.classicalmidi.co.uk/moszk.htm']
sources['classical']['mozart']       = ['http://www.midiworld.com/mozart.htm','http://www.classicalmidi.co.uk/mozart.htm']
sources['classical']['nikolaievich'] = ['http://www.classicalmidi.co.uk/scab.htm']
sources['classical']['orff']         = ['http://www.classicalmidi.co.uk/orff.htm']
sources['classical']['pachelbel']    = ['http://www.classicalmidi.co.uk/pach.htm']
sources['classical']['paderewski']   = ['http://www.classicalmidi.co.uk/paderewski.htm']
sources['classical']['pagg']         = ['http://www.classicalmidi.co.uk/pagg.htm']
sources['classical']['palestrina']   = ['http://www.classicalmidi.co.uk/palestrina.htm']
sources['classical']['paradisi']     = ['http://www.classicalmidi.co.uk/paradisi.htm']
sources['classical']['poulenc']      = ['http://www.classicalmidi.co.uk/poulenc.htm']
sources['classical']['pres']         = ['http://www.classicalmidi.co.uk/pres.htm']
sources['classical']['prokif']       = ['http://www.classicalmidi.co.uk/prokif.htm']
sources['classical']['puccini']      = ['http://www.classicalmidi.co.uk/puccini.htm']
sources['classical']['rachmaninov']  = ['http://www.midiworld.com/rachmaninov.htm','http://www.classicalmidi.co.uk/rach.htm']
sources['classical']['ravel']        = ['http://www.classicalmidi.co.uk/ravel1.htm']
sources['classical']['respig']       = ['http://www.classicalmidi.co.uk/respig.htm']
sources['classical']['rimsky']       = ['http://www.classicalmidi.co.uk/rimsky.htm']
sources['classical']['rossini']      = ['http://www.classicalmidi.co.uk/rossini.htm']
sources['classical']['strauss']     = ['http://www.classicalmidi.co.uk/rstrauss.htm']
sources['classical']['sacrlatt']     = ['http://www.classicalmidi.co.uk/sacrlatt.htm']
sources['classical']['saens']        = ['http://www.classicalmidi.co.uk/saens.htm']
sources['classical']['sanz']         = ['http://www.classicalmidi.co.uk/sanz.htm']
sources['classical']['satie']        = ['http://www.classicalmidi.co.uk/satie.htm']
sources['classical']['scarlatti']    = ['http://www.midiworld.com/scarlatti.htm','http://www.classicalmidi.co.uk/scarlatt.htm']
sources['classical']['schoberg']     = ['http://www.classicalmidi.co.uk/schoberg.htm']
sources['classical']['schubert']     = ['http://www.classicalmidi.co.uk/schubert.htm']
sources['classical']['schumann']     = ['http://www.midiworld.com/schumann.htm', 'http://www.classicalmidi.co.uk/schuman.htm']
sources['classical']['scriabin']     = ['http://www.midiworld.com/scriabin.htm']
sources['classical']['shostakovich'] = ['http://www.classicalmidi.co.uk/shost.htm']
sources['classical']['sibelius']     = ['http://www.classicalmidi.co.uk/sibelius.htm']
sources['classical']['soler']        = ['http://www.classicalmidi.co.uk/soler.htm']
sources['classical']['sor']          = ['http://www.classicalmidi.co.uk/sor.htm']
sources['classical']['sousa']        = ['http://www.classicalmidi.co.uk/sousa.htm']
sources['classical']['stravinsky']   = ['http://www.classicalmidi.co.uk/strav.htm']
sources['classical']['sullivan']     = ['http://www.classicalmidi.co.uk/sull.htm']
sources['classical']['susato']       = ['http://www.classicalmidi.co.uk/susato.htm']
sources['classical']['taktak']       = ['http://www.classicalmidi.co.uk/taktak.htm']
sources['classical']['taylor']       = ['http://www.classicalmidi.co.uk/taylor.htm']
sources['classical']['tchaikovsky']  = ['http://www.midiworld.com/tchaikovsky.htm','http://www.classicalmidi.co.uk/tch.htm']
sources['classical']['thomas']       = ['http://www.classicalmidi.co.uk/thomas.htm']
sources['classical']['vaughan']      = ['http://www.classicalmidi.co.uk/vaughan.htm']
sources['classical']['verdi']        = ['http://www.classicalmidi.co.uk/verdi.htm']
sources['classical']['vivaldi']      = ['http://www.classicalmidi.co.uk/vivaldi.htm']
sources['classical']['wagner']       = ['http://www.classicalmidi.co.uk/wagner.htm']
sources['classical']['walton']       = ['http://www.classicalmidi.co.uk/walton.htm']
sources['classical']['wolf']         = ['http://www.classicalmidi.co.uk/wolf.htm']
sources['classical']['wyschnegradsky'] = ['http://www.classicalmidi.co.uk/Wyschnegradsky.htm']
sources['classical']['yradier']      = ['http://www.classicalmidi.co.uk/yradier.htm']


#sources['classical']['misc']         = ['http://www.midiworld.com/classic.htm']
##sources['classical']['dukas']        = ['http://www.classicalmidi.co.uk/dukas.htm']
#sources['classical']['earlymusic']   = ['http://www.midiworld.com/earlymus.htm']


ignore_patterns                      = ['xoom']

file_list = {}

file_list['validation'] = ['classical/byrd/byrd17.mid', \
'classical/handel/Mes03.mid', \
'classical/bach/430bjsbmm15.mid', \
'classical/beethoven/1472beetpc5mov3.mid', \
'classical/messager/2595mirette04.mid', 'classical/mendelssohn/Eli09.mid', \
'classical/liszt/2074hungarian.mid', \
'classical/vaughan/612march2Steven.mid', \
'classical/chopin/mazrka13.mid', \
'classical/albenizmateo/albenizmateosonatazapateadop5t.mid', \
'classical/handel/va-tacit.mid', \
'classical/carulli/843guitar.mid', \
'classical/mendelssohn/mnsnt13.mid', \
'classical/bach/bwv816.mid', \
'classical/bach/1684tpvent7w.mid', \
'classical/lemire/1205ranali.mid', \
'classical/bach/bwv794.mid', \
'classical/beethoven/pathet3.mid', \
'classical/mendelssohn/men-gon1.mid', \
'classical/handel/Mes08.mid', \
'classical/liszt/mephist2.mid', \
'classical/albenizisaac/1153albzse05.mid', \
'classical/bach/vp1-6sad.mid', \
'classical/bach/bwv772.mid', \
'classical/handel/mes19.mid', \
'classical/bach/bwv861.mid', \
'classical/handel/gp_fwork.mid', \
'classical/elgar/1745ee1416.mid', \
'classical/bach/cs5-2all.mid', \
'classical/mozart/mfig.mid', \
'classical/kuhlau/1070op20no2.mid', \
'classical/haydn/2118n7wrdsp11.mid', \
'classical/handel/thesoft.mid', \
'classical/prokif/2002wolf.mid', \
'classical/rachmaninov/pgvr1115.mid', \
'classical/brahms/402op34b-2.mid', \
'classical/haydn/hayln07.mid', \
'classical/prokif/408prokofie.mid', \
'classical/barber/1812excurno4.mid', \
'classical/sullivan/2548haddon02d03.mid', \
'classical/handel/vadoro.mid', \
'classical/mozart/57911meno10.mid', \
'classical/foster/suzan.mid', \
'classical/rimsky/696gpschez.mid', \
'classical/shostakovich/5653rdshos.mid', \
'classical/hummel/hmwoo23b.mid', \
'classical/chopin/2193song19.mid', \
'classical/mendelssohn/1959sng67232.mid', \
'classical/schubert/2587momentsmusicaux1.mid', \
'classical/chopin/mazrka25.mid', \
'classical/haydn/413h641.mid', \
'classical/byrd/byrd20.mid', \
'classical/mozart/mozk622c.mid', \
'classical/chopin/2412chopinprelude.mid', \
'classical/karg/70inter1.mid', \
'classical/czerny/983czerny12Steven.mid', \
'classical/dussek/2279laconsolSteven.mid', \
'classical/byrd/byrd26.mid', \
'classical/mozart/2025pianos3no2.mid', \
'classical/bach/bwv791.mid', \
'classical/vaughan/691rondoSteven.mid', \
'classical/beethoven/2232bagat7Steven.mid', \
'classical/haydn/hay-p33c.mid', \
'classical/german/919quartSteven.mid', \
'classical/sullivan/18BellChorus.mid', \
'classical/mendelssohn/2207fugemino1.mid', \
'classical/moszkowski/951presSteven.mid', \
'classical/griff/804grifo7n3.mid', \
'classical/bach/bwv782.mid', \
'classical/busoni/2242bsfanjsb.mid', \
'classical/bach/bwv847.mid', \
'classical/hummel/mthmwo4c.mid', \
'classical/messager/2611mirette25.mid', \
'classical/dvorak/18_8211.mid', \
'classical/handel/stcmarch.mid', \
'classical/lobos/147prelud1.mid', \
'classical/bouwer/342brow.mid', \
'classical/schumann/sch-fan2.mid', \
'classical/saens/39cl36xg.mid', \
'classical/tchaikovsky/12arab.mid', \
'classical/poulenc/1822lecombledistinction.mid', \
'classical/beethoven/138.mid', \
'classical/karg/71inter2.mid', \
'classical/corelli/2015corelno4.mid', \
'classical/schubert/1707sylvia.mid', \
'classical/mozart/2917requm2.mid', \
'classical/grieg/2316tmgr54no5.mid', \
'classical/bach/bwv779.mid', \
'classical/messager/2907mirette19.mid', \
'classical/tchaikovsky/tchpc12.mid', \
'classical/bergmuller/2510bl109n09.mid', \
'classical/bach/cs1-6gig.mid', \
'classical/dvorak/378dvse1.mid', \
'classical/byrd/byrd90.mid', \
'classical/byrd/byrd62.mid', \
'classical/brahms/24brahmspf.mid', \
'classical/bach/1717tpvent14w.mid', \
'classical/kuhlau/1076quartet4.mid', \
'classical/palestrina/1877voimiponeste.mid', \
'classical/scriabin/maz25n3.mid', \
'classical/faure/2708faure.mid', \
'classical/german/830choruSteven.mid', \
'classical/mendelssohn/mndnoct.mid', \
'classical/barber/barberadagiostrings.mid', \
'classical/maier/atalan8.mid', \
'classical/rossini/2671rossiniladanza.mid', \
'classical/hummel/mthmop18.mid', \
'classical/czerny/982czerny11Steven.mid', \
'classical/soler/597sonata84.mid', \
'classical/hummel/taoh24e4.mid', \
'classical/handel/Mes53a.mid', \
'classical/byrd/byrd33.mid', \
'classical/moszkowski/2371mser.mid', \
'classical/anderson/520blue.mid', \
'classical/karg/1322alaburla.mid', \
'classical/brahms/403op34b-3.mid', \
'classical/clementi/2388tmcl36no3.mid', \
'classical/scriabin/pr37n2.mid', \
'classical/handel/cara.mid', \
'classical/janacek/1211jansinf4.mid', \
'classical/bach/vp3-3gav.mid', \
'classical/palestrina/2158gpagnsd2.mid', \
'classical/liszt/mephist1.mid', \
'classical/maier/atala43.mid', \
'classical/maier/atalan7.mid', \
'classical/rachmaninov/386pagvar18.mid', \
'classical/mehul/1653mhchasse.mid', \
'classical/bartok/btkconc2.mid', \
'classical/sanz/1300gs5zarab.mid', \
'classical/sullivan/2582chieftain211.mid', \
'classical/mendelssohn/ital-4.mid', \
'classical/bach/1668bachvari.mid', \
'classical/curzon/2267gprh2.mid', \
'classical/mozart/2294rondomozSteven.mid', \
'classical/rachmaninov/rach_op30_1.mid', \
'classical/hummel/mthmvr75.mid', \
'classical/delius/473serenade3.mid', \
'classical/bach/stmatt.mid', \
'classical/bach/bwv826.mid', \
'classical/lobos/1777bs2.mid', \
'classical/chopin/chno1503.mid', \
'classical/byrd/1734no58carmanswhistle.mid', \
'classical/brahms/1359brawal10.mid', \
'classical/boccherini/499boccher.mid', \
'classical/cramer/2335cr39d.mid', \
'classical/sullivan/2555haddon14.mid', \
'classical/berlioz/1110harold3.mid', \
'classical/bach/bwv966.mid', \
'classical/chopin/199etude25-1.mid', \
'classical/schumann/806schumm4.mid', \
'classical/vaughan/593mountSteven.mid', \
'classical/sibelius/2288aspen3.mid', \
'classical/wagner/1494rienziov.mid', \
'classical/beethoven/2355rondo51n2Steven.mid', \
'classical/schubert/2724schubert4hand.mid', \
'classical/nikolaievich/1091maz241.mid', \
'classical/lobos/788caicaibalao.mid', \
'classical/bartok/2497tm10es08.mid', \
'classical/bach/cs4-6gig.mid', \
'classical/mozart/2249phantasieSteven.mid', \
'classical/respig/1586rf.mid', \
'classical/ravel/330Grecque3.mid', \
'classical/scriabin/pr37n1.mid', \
'classical/bach/2653sicilianos766.mid', \
'classical/bach/brand41s.mid', \
'classical/sibelius/1438jsop105.mid', \
'classical/schumann/sr12-2.mid', \
'classical/coates/3004starofgod.mid', \
'classical/handel/amin_b.mid', \
'classical/bach/1699tpinven8.mid', \
'classical/sor/1797sorstudyno5.mid', \
'classical/haydn/2278fantasiaSteven.mid', \
'classical/schumann/sr15-12.mid', \
'classical/bach/vp1-4cod.mid', \
'classical/nikolaievich/1085maz01.mid', \
'classical/liszt/lszt_vi.mid', \
'classical/franck/1772chorale1e.mid', \
'classical/pachelbel/1139pbgigue.mid', \
'classical/bach/3411006_lou.mid', \
'classical/vaughan/1505slowSteven.mid', \
'classical/schumann/sr28-2.mid', \
'classical/brahms/haydnvar.mid', \
'classical/albenizisaac/albesp6.mid', \
'classical/bizet/568levoici.mid', \
'classical/coates/2932princessofthedawn.mid', \
'classical/rossini/Dominedeus.mid', \
'classical/handel/hans6a.mid', \
'classical/brahms/waltz_14.mid', \
'classical/hummel/humm_t2.mid', \
'classical/handel/zadok1.mid', \
'classical/schumann/scenes.mid', \
'classical/bartok/kinder5.mid', \
'classical/bartok/mikro126.mid', \
'classical/german/1000jelfSteven.mid', \
'classical/czerny/1008czerny13Steven.mid', \
'classical/ginast/457ginson1.mid', \
'classical/hummel/mthmwo3a.mid', \
'classical/sullivan/23SongGama.mid', \
'classical/chopin/mazrka49.mid', \
'classical/handel/sharp.mid', \
'classical/sullivan/2576chieftain205.mid', \
'classical/bergmuller/1191burgmul8Steven.mid', \
'classical/clementi/1830sonatin36Steven.mid', \
'classical/bach/bwv1060c.mid', \
'classical/handel/mes18.mid', \
'classical/taylor/1486demandSteven.mid', \
'classical/czerny/37cz740.mid', \
'classical/scriabin/scrsym32.mid', \
'classical/brahms/waltz_15.mid', \
'classical/mendelssohn/mendel.mid', \
'classical/mozart/jm_mozdi.mid', \
'classical/bach/cs4-1pre.mid', \
'classical/hummel/1574hmcello2.mid', \
'classical/byrd/byrd41.mid', \
'classical/albenizisaac/albesp5.mid', \
'classical/mozart/2026pianos3no3.mid', \
'classical/maier/atala13.mid', \
'classical/mozart/mozk216c.mid', \
'classical/satie/1748satogv3.mid', \
'classical/bach/bjsbmm16.mid', \
'classical/bizet/2650preludeno1of2.mid', \
'classical/sibelius/1440jsop22m2.mid', \
'classical/handel/father.mid', \
'classical/nikolaievich/315scriab06.mid', \
'classical/handel/hallujah.mid', \
'classical/haydn/1650haydn942.mid', \
'classical/holst/764jig.mid', \
'classical/tchaikovsky/nutcrkr4.mid', \
'classical/chopin/mazrka33.mid', \
'classical/boccherini/411boccq1.mid', \
'classical/mendelssohn/Eli42a.mid', \
'classical/rachmaninov/517rachprec.mid', \
'classical/sullivan/21OpeningChorus.mid', \
'classical/liszt/1558earlking.mid', \
'classical/scarlatti/2277sonate345.mid', \
'classical/mozart/mozk246a.mid', \
'classical/mendelssohn/1850mendel.mid', \
'classical/mozart/mozk333b.mid', \
'classical/lenar/veuvj.mid', \
'classical/holst/470firstsuitemv1.mid', \
'classical/handel/bourree.mid', \
'classical/griff/802grifo5n2.mid', \
'classical/franck/1769chasseur.mid', \
'classical/dupre/1512prel3.mid', \
'classical/beethoven/1744bps32no2.mid', \
'classical/shostakovich/2659tmsk1302.mid', \
'classical/bach/brand5.mid', \
'classical/chopin/2186song12.mid', \
'classical/hummel/mthmtrc1.mid', \
'classical/scarlatti/gpk519.mid', \
'classical/sibelius/1437jsop104m4.mid', \
'classical/handel/hans6b.mid', \
'classical/liszt/lszt_et3.mid', \
'classical/liszt/9.mid', \
'classical/joplin/828jentrtnr.mid', \
'classical/bartok/bear-dnc.mid', \
'classical/liszt/1557danteson.mid', \
'classical/rachmaninov/383pagavar7.mid', \
'classical/messager/2593mirette02.mid', \
'classical/tchaikovsky/704gp3nut.mid', \
'classical/maier/atala14.mid', \
'classical/bach/bwv785.mid', \
'classical/faure/2194gpsicil.mid', \
'classical/brahms/hung3.mid', \
'classical/mozart/mozk175b.mid', \
'classical/soler/1474soler91a.mid', \
'classical/mendelssohn/Eli38.mid', \
'classical/bach/kommsues.mid', \
'classical/handel/op6n08m4.mid', \
'classical/hummel/728mthuso1a.mid', \
'classical/scriabin/pr37n3.mid', \
'classical/handel/op6n04m4.mid', \
'classical/dupre/1513synpass1.mid', \
'classical/bergmuller/2507bl109n06.mid', \
'classical/soler/1481soler92d.mid', \
'classical/bach/bwv903.mid', \
'classical/moszkowski/19.mid', \
'classical/gilbert/166.mid', \
'classical/hummel/humop101.mid', \
'classical/beethoven/135.mid', \
'classical/bartok/1309barmspc4.mid', \
'classical/handel/op6n10m1.mid', \
'classical/byrd/byrd87.mid', \
'classical/handel/op6n08m2.mid', \
'classical/byrd/byrd46.mid', \
'classical/gershwin/2131prgybess.mid', \
'classical/chopin/chet1005.mid', \
'classical/thomas/1833mignon2.mid', \
'classical/chopin/chet1003.mid', \
'classical/gott/1240ballade6.mid', \
'classical/albinoni/1038albimag.mid', \
'classical/prokif/2810gpdkt.mid', \
'classical/bergmuller/burgmullerop100n24.mid', \
'classical/bizet/1428lIntermezzo.mid', \
'classical/holst/HolstVenusfixed.mid', \
'classical/byrd/byrd27.mid', \
'classical/sullivan/2578chieftain207.mid', \
'classical/scriabin/scripr17.mid', \
'classical/kuhlau/1072op55no1.mid', \
'classical/mendelssohn/msw34.mid', \
'classical/sibelius/2286roantree1.mid', \
'classical/bach/air.mid', \
'classical/handel/hanarvar.mid', \
'classical/bach/bwv858.mid', \
'classical/debussy/320Egyptian.mid', \
'classical/grieg/Pergynt4.mid', \
'classical/handel/raging.mid', \
'classical/mozart/mozk488a.mid', \
'classical/mozart/mozk218c.mid', \
'classical/handel/wewill.mid', \
'classical/grieg/1856pianoconcertop16no2.mid']

file_list['test'] = ['classical/mozart/245div1.mid', \
'classical/byrd/1731fantasia2.mid', \
'classical/mozart/mozk333a.mid', \
'classical/puccini/155.mid', \
'classical/nikolaievich/2096maz25n9.mid', \
'classical/grain/850gumsuckr.mid', \
'classical/brahms/1369brawalz6.mid', \
'classical/hummel/hummel3.mid', \
'classical/messager/2598mirette07.mid', \
'classical/ravel/331Grecque4.mid', \
'classical/garoto/343des.mid', \
'classical/haydn/hay-p32c.mid', \
'classical/handel/lofty.mid', \
'classical/palestrina/1876almaredemptoris.mid', \
'classical/bach/1705tpinven14.mid', \
'classical/borodin/2312tmaucouv.mid', \
'classical/coates/gpwork.mid', \
'classical/stravinsky/1622spring.mid', \
'classical/bartok/arobusto.mid', \
'classical/modest/834gpgpak.mid', \
'classical/beethoven/beets3m2.mid', \
'classical/nikolaievich/1089maz17.mid', \
'classical/byrd/byrd75.mid', \
'classical/handel/op6n06m4.mid', \
'classical/cherubini/1117cherubi1.mid', \
'classical/sor/1798sorstudyno6.mid', \
'classical/handel/miocaro.mid', \
'classical/stravinsky/1288fbirdInfdance.mid', \
'classical/faure/1767faurecan.mid', \
'classical/bellini/1913absent.mid', \
'classical/brahms/2428schmucke1.mid', \
'classical/bach/bwv976.mid', \
'classical/sibelius/1441jsop25m1.mid', \
'classical/chopin/mazrka36.mid', \
'classical/bergmuller/2511bl109n10.mid', \
'classical/mendelssohn/Eli16.mid', \
'classical/handel/flute-4.mid', \
'classical/bruckner/248brukoar.mid', \
'classical/dupre/1508n15pn02.mid', \
'classical/bartok/kinder1.mid', \
'classical/grieg/2319tmgrelve.mid', \
'classical/poulenc/773poulsf.mid', \
'classical/sor/1804sorop1no5.mid', \
'classical/chopin/mazrka28.mid', \
'classical/rossini/1182rossinbarsevil4.mid', \
'classical/mendelssohn/meneli38.mid', \
'classical/bach/bwv913.mid', \
'classical/byrd/byrd31.mid', \
'classical/hummel/mthuso1b.mid', \
'classical/bartok/mikro131.mid', \
'classical/griff/801grifo7n1.mid', \
'classical/bach/1685tpvent8w.mid', \
'classical/haydn/hayln10.mid', \
'classical/byrd/byrd28.mid', \
'classical/bach/bwv980.mid', \
'classical/handel/op6n09m3.mid', \
'classical/hummel/mthuso1a.mid', \
'classical/chopin/chet2511.mid', \
'classical/bach/1893s1cour.mid', \
'classical/bach/bwv915.mid', \
'classical/brahms/167.mid', \
'classical/bach/bjsbmm14.mid', \
'classical/mozart/mozk299a.mid', \
'classical/handel/op6n06m2.mid', \
'classical/chopin/2185song11.mid', \
'classical/shostakovich/2666tmsk1309.mid', \
'classical/hummel/humm81m3.mid', \
'classical/dvorak/107.mid', \
'classical/handel/blest.mid', \
'classical/bach/bwv827.mid', \
'classical/prokif/2352tmprktoc.mid', \
'classical/poulenc/553poulenc1pastorale.mid', \
'classical/german/2359tomjones.mid', \
'classical/chopin/2187song13.mid', \
'classical/franck/1770choraleamin.mid', \
'classical/bach/bwv792.mid', \
'classical/handel/mes23.mid', \
'classical/clementi/2387tmcl36no2.mid', \
'classical/handel/Mes52.mid', \
'classical/vivaldi/1532avautunno1.mid', \
'classical/faure/2409fauromance.mid', \
'classical/bach/bwv979.mid', \
'classical/brahms/1367brawalz4.mid', \
'classical/schubert/48d958-3.mid', \
'classical/cramer/2338cr20d.mid', \
'classical/handel/han4-5d.mid', \
'classical/alkan/1173nuitdete.mid', \
'classical/handel/op6n01m1.mid', \
'classical/bartok/mikro151.mid', \
'classical/hummel/humm81m1.mid', \
'classical/joplin/173.mid', \
'classical/stravinsky/1620symno2.mid', \
'classical/scriabin/scrsym31.mid', \
'classical/liszt/33.mid', \
'classical/beethoven/2212bagat6Steven.mid', \
'classical/ibert/681aneblanc.mid', \
'classical/handel/op6n11m1.mid', \
'classical/mozart/4321_duett.mid', \
'classical/handel/clangour.mid', \
'classical/bach/654gpbad.mid', \
'classical/bach/1840brand2no3.mid', \
'classical/byrd/wbpavan.mid', \
'classical/mendelssohn/Eli39.mid', \
'classical/brahms/waltz_04.mid', \
'classical/bach/106.mid', \
'classical/czerny/953czerny10Steven.mid', \
'classical/brahms/waltz_05.mid', \
'classical/handel/stc-over.mid', \
'classical/clementi/2386tmcl36no1.mid', \
'classical/schumann/sr15-7.mid', \
'classical/chopin/mazrka46.mid', \
'classical/lemire/1203filonor.mid', \
'classical/beethoven/142.mid', \
'classical/byrd/byrd10.mid', \
'classical/chopin/chpson3b.mid', \
'classical/handel/op6n09m5.mid', \
'classical/bartok/mikro130.mid', \
'classical/hummel/1955humt2no3.mid', \
'classical/german/1764song3.mid', \
'classical/handel/op6n06gm.mid', \
'classical/hummel/hummop23.mid', \
'classical/haydn/hdn104_4.mid', \
'classical/handel/Mes33.mid', \
'classical/bach/bwv790.mid', \
'classical/chopin/mazrka04.mid', \
'classical/bach/2439tmtoccfg.mid', \
'classical/chopin/mazrka34.mid', \
'classical/verdi/1605no03brind.mid', \
'classical/maier/atala39.mid', \
'classical/mozart/2344serenadeinbb.mid', \
'classical/respig/840respigh4.mid', \
'classical/byrd/byrd84.mid', \
'classical/gershwin/2221americanparis.mid', \
'classical/hummel/hmmtop95.mid', \
'classical/pachelbel/1585pchciacd.mid', \
'classical/beethoven/beets3m3.mid', \
'classical/stravinsky/1629riteno1g.mid', \
'classical/mendelssohn/238fugh-a.mid', \
'classical/messia/90.mid', \
'classical/poulenc/1827poultdp4.mid', \
'classical/liszt/1561etude10.mid', \
'classical/mendelssohn/4604fugem.mid', \
'classical/brahms/bvhi.mid', \
'classical/schumann/sr15-11.mid', \
'classical/gershwin/2678necessar.mid', \
'classical/haydn/26.mid', \
'classical/beethoven/beeth3_2.mid', \
'classical/handel/op6n07m3.mid', \
'classical/bach/trioson2.mid', \
'classical/sibelius/2290spruce5.mid', \
'classical/busoni/2243busonicf.mid', \
'classical/rimsky/1669bee3.mid', \
'classical/handel/flute-5.mid', \
'classical/soler/1473soler90.mid', \
'classical/mozart/495p141ck.mid']


class MusicDataLoader(object):

  def __init__(self, datadir, select_validation_percentage, select_test_percentage, works_per_composer=None, pace_events=False, synthetic=None, tones_per_cell=1, single_composer=None):
    self.datadir = datadir
    self.output_ticks_per_quarter_note = 384.0
    self.tones_per_cell = tones_per_cell
    self.single_composer = single_composer
    self.pointer = {}
    self.pointer['validation'] = 0
    self.pointer['test'] = 0
    self.pointer['train'] = 0
    if synthetic == 'chords':
      self.generate_chords(pace_events=pace_events)
    elif not datadir is None:
      print ('Data loader: datadir: {}'.format(datadir))
      self.download_midi_data()
      self.read_data(select_validation_percentage, select_test_percentage, works_per_composer, pace_events)

  def download_midi_data(self):
    """
    download_midi_data will download a number of midi files, linked from the html
    pages specified in the sources dict, into datadir. There will be one subdir
    per genre, and within each genre-subdir, there will be a subdir per composer.
    Hence, similar to the structure of the sources dict.
    """
    midi_files = {}
    print("midi data")

    if os.path.exists(os.path.join(self.datadir, 'do-not-redownload.txt')):
      print ( 'Already completely downloaded, delete do-not-redownload.txt to check for files to download.')
      return
    for genre in sources:
      midi_files[genre] = {}
      for composer in sources[genre]:
        midi_files[genre][composer] = []
        for url in sources[genre][composer]:
          print ("url", url)
          try:
              response = urlopen(url)
              data = response.read().decode('latin-1')
          except Exception as r:
              print('error',r)
              continue
          #if 'classicalmidi' in url:
          #  headers = response.info()
          #  print ( headers

          #htmlinks = re.findall('"(  ?[^"]+\.htm)"', data)
          #for link in htmlinks:
          #  print ( 'http://www.classicalmidi.co.uk/'+strip(link)
          
          # make urls absolute:
          urlparsed = urlparse(url)
          
          data = re.sub('href="\/', 'href="http://'+urlparsed.hostname+'/', data, flags= re.IGNORECASE)
          data = re.sub('href="(?!http:)', 'href="http://'+urlparsed.hostname+urlparsed.path[:urlparsed.path.rfind('/')]+'/', data, flags= re.IGNORECASE)
          #if 'classicalmidi' in url:
          #  print ( data
          
          links = re.findall('"(http://[^"]+\.mid)"', data)
          for link in links:
            cont = False
            for p in ignore_patterns:
              if p in link:
                print ( 'Not downloading links with {}'.format(p))
                cont = True
                continue
            if cont: continue
            print ( link)
            filename = link.split('/')[-1]
            valid_chars = "-_.()%s%s" % (string.ascii_letters, string.digits)
            filename = ''.join(c for c in filename if c in valid_chars)
            print ( genre+'/'+composer+'/'+filename)
            midi_files[genre][composer].append(filename)
            localdir = os.path.join(os.path.join(self.datadir, genre), composer)
            localpath = os.path.join(localdir, filename)
            if os.path.exists(localpath):
              print ( 'File exists. Not redownloading: {}'.format(localpath))
            else:
              try:
                response_midi = urlopen(link)
                try: os.makedirs(localdir)
                except: pass
                data_midi = response_midi.read()
                #print(type(data_midi))
                
                #if 'DOCTYPE html PUBLIC' in data_midi:
                #  print ( 'Seems to have been served an html page instead of a midi file. Continuing with next file.')
                #elif 'RIFF' in data_midi[0:9]:
                #  print ( 'Seems to have been served an RIFF file instead of a midi file. Continuing with next file.')
                #else:
                with open(localpath, 'wb') as f:
                    f.write(data_midi)
              except Exception as e:
                print("error 2\n",e)
                #response_midi = urlopen(link)
                #print(response_midi.read())
                print ( 'Failed to fetch {}'.format(link))
    with open(os.path.join(self.datadir, 'do-not-redownload.txt'), 'w') as f:
      f.write('This directory is considered completely downloaded.')

  def generate_chords(self, pace_events):
    """
    generate_chords generates synthetic songs with either major or minor chords
    in a chosen scale.

    returns a list of tuples, [genre, composer, song_data]
    Also saves this list in self.songs.

    Time steps will be fractions of beat notes (32th notes).
    """

    self.genres = ['classical']
    print (('num genres:{}'.format(len(self.genres))))
    self.composers = ['generated_chords']
    print (('num composers: {}'.format(len(self.composers))))

    self.songs = {}
    self.songs['validation'] = []
    self.songs['test'] = []
    self.songs['train'] = []
    
    # https://songwritingandrecordingtips.wordpress.com/2012/02/09/chord-progressions-that-fit-together/
    # M m m M M m
    base_tones     = [0,2,4,5,7,9]
    chord_is_major = [True,False,False,True,True,False]
    #(W-W-H-W-W-W-H)
    #(2 2 1 2 2 2 1)
    major_third_offset = 4
    minor_third_offset = 3
    fifth_offset       = 7

    songlength = 500
    numsongs = 1000

    genre = self.genres[0]
    composer = self.composers[0]
    
    #write_files = False
    #print (('write_files = False')
    #if self.datadir is not None:
    #  write_files = True
    #  print (('write_files = True')
    #  dirnameforallfiles = os.path.join(self.datadir, os.path.join(genre, composer))
    #  if not os.path.exists(dirnameforallfiles):
    #    os.makedirs(dirnameforallfiles)
    #  else:
    #    print (('write_files = False')
    #    write_files = False

    for i in range(numsongs):
      # OVERFIT
      if i % 100 == 99:
        print ( 'Generating songs {}/{}: {}'.format(genre, composer, (i+1)))
      
      song_data = []
      key = random.randint(0,100)
      #key = 50

      # Tempo:
      ticks_per_quarter_note = 384
      
      for j in range(songlength):
        last_event_input_tick=0
        not_closed_notes = []
        begin_tick = float(j*ticks_per_quarter_note)
        velocity = float(100)
        length = ticks_per_quarter_note-1
        
        # randomness out of chords that 'fit'
        # https://songwritingandrecordingtips.wordpress.com/2012/02/09/chord-progressions-that-fit-together/
        base_tone_index = random.randint(0,5)
        base_tone = key+base_tones[base_tone_index]
        is_major = chord_is_major[base_tone_index]
        third = base_tone+major_third_offset
        if not is_major:
          third = base_tone+minor_third_offset
        fifth = base_tone+fifth_offset

        note = [0.0]*(NUM_FEATURES_PER_TONE+1)
        note[LENGTH]     = length
        note[FREQ]       = tone_to_freq(base_tone)
        note[VELOCITY]   = velocity
        note[BEGIN_TICK] = begin_tick
        song_data.append(note)
        note2 = note[:]
        note2[FREQ] = tone_to_freq(third)
        song_data.append(note2)
        note3 = note[:]
        note3[FREQ] = tone_to_freq(fifth)
        song_data.append(note3)
      song_data.sort(key=lambda e: e[BEGIN_TICK])
      #print ((song_data)
      #sys.exit()
      if (pace_events):
        pace_event_list = []
        pace_tick = 0.0
        song_tick_length = song_data[-1][BEGIN_TICK]+song_data[-1][LENGTH]
        while pace_tick < song_tick_length:
          song_data.append([0.0, 440.0, 0.0, pace_tick, 0.0])
          pace_tick += float(ticks_per_quarter_note)/input_ticks_per_output_tick
        song_data.sort(key=lambda e: e[BEGIN_TICK])
      if self.datadir is not None and i==0:
        filename = os.path.join(self.datadir, '{}.mid'.format(i))
        if not os.path.exists(filename):
          print (('saving: {}.'.format(filename)))
          self.save_data(filename, song_data)
        else:
          print (('file exists. Not overwriting: {}.'.format(filename)))
      
      if i%100 == 0:
        self.songs['validation'].append([genre, composer, song_data])
      elif i%100 == 1:
        self.songs['test'].append([genre, composer, song_data])
      else:
        self.songs['train'].append([genre, composer, song_data])
    
    self.pointer['validation'] = 0
    self.pointer['test'] = 0
    self.pointer['train'] = 0
    print (('lens: train: {}, val: {}, test: {}'.format(len(self.songs['train']), len(self.songs['validation']), len(self.songs['test']))))
    return self.songs


  def read_data(self, select_validation_percentage, select_test_percentage, works_per_composer, pace_events):
    """
    read_data takes a datadir with genre subdirs, and composer subsubdirs
    containing midi files, reads them into training data for an rnn-gan model.
    Midi music information will be real-valued frequencies of the
    tones, and intensity taken from the velocity information in
    the midi files.

    returns a list of tuples, [genre, composer, song_data]
    Also saves this list in self.songs.

    Time steps will be fractions of beat notes (32th notes).
    """

    self.genres = sorted(sources.keys())
    print (('num genres:{}'.format(len(self.genres))))
    if self.single_composer is not None:
      self.composers = [self.single_composer]
    else:
      self.composers = []
      for genre in self.genres:
        self.composers.extend(sources[genre].keys())
      if debug == 'overfit':
        self.composers = self.composers[0:1]
      self.composers = list(set(self.composers))
      self.composers.sort()
    print (('num composers: {}'.format(len(self.composers))))
    print (('limit works per composer: {}'.format(works_per_composer)))

    self.songs = {}
    self.songs['validation'] = []
    self.songs['test'] = []
    self.songs['train'] = []

    #max_composers = 2
    #composer_id   = 0
    if select_validation_percentage or select_test_percentage:
      filelist = []
      for genre in self.genres:
        for composer in self.composers:
          current_path = os.path.join(self.datadir,os.path.join(genre, composer))
          if not os.path.exists(current_path):
            print ( 'Path does not exist: {}'.format(current_path))
            continue
          files = os.listdir(current_path)
          works_read = 0
          for i,f in enumerate(files):
            if os.path.isfile(os.path.join(current_path,f)):
              print (('appending {}'.format(os.path.join(os.path.join(genre, composer), f))))
              filelist.append(os.path.join(os.path.join(genre, composer), f))
              works_read += 1
              if works_per_composer is not None and works_read >= works_per_composer:
                break
      print (('filelist len: {}'.format(len(filelist))))
      random.shuffle(filelist)
      print (('filelist len: {}'.format(len(filelist))))
      
      validation_len = 0
      if select_test_percentage:
        validation_len = int(float(select_validation_percentage/100.0)*len(filelist))
        print (('validation len: {}'.format(validation_len)))
        file_list['validation'] = filelist[0:validation_len]
        print ( ('Selected validation set (FLAG --select_validation_percentage): {}'.format(file_list['validation'])))
      if select_test_percentage:
        test_len = int(float(select_test_percentage/100.0)*len(filelist))
        print (('test len: {}'.format(test_len)))
        file_list['test'] = filelist[validation_len:validation_len+test_len]
        print ( ('Selected test set (FLAG --select_test_percentage): {}'.format(file_list['test'])))
    
    # OVERFIT
    count = 0

    for genre in self.genres:
      # OVERFIT
      if debug == 'overfit' and count > 20: break
      for composer in self.composers:
        # OVERFIT
        if debug == 'overfit' and composer not in self.composers: continue
        if debug == 'overfit' and count > 20: break
        current_path = os.path.join(self.datadir,os.path.join(genre, composer))
        if not os.path.exists(current_path):
          print ( 'Path does not exist: {}'.format(current_path))
          continue
        files = os.listdir(current_path)
        #composer_id += 1
        #if composer_id > max_composers:
        #  print (('Only using {} composers.'.format(max_composers))
        #  break
        for i,f in enumerate(files):
          # OVERFIT
          if debug == 'overfit' and count > 20: break
          count += 1
          
          if works_per_composer is not None and i >= works_per_composer:
            break
          
          if i % 100 == 99 or i+1 == len(files) or i+1 == works_per_composer:
            print ( 'Reading files {}/{}: {}'.format(genre, composer, (i+1)))
          if os.path.isfile(os.path.join(current_path,f)):
            song_data = self.read_one_file(current_path, f, pace_events)
            if song_data is None:
              continue
            if os.path.join(os.path.join(genre, composer), f) in file_list['validation']:
              self.songs['validation'].append([genre, composer, song_data])
            elif os.path.join(os.path.join(genre, composer), f) in file_list['test']:
              self.songs['test'].append([genre, composer, song_data])
            else:
              self.songs['train'].append([genre, composer, song_data])
          #b0reak
    random.shuffle(self.songs['train'])
    self.pointer['validation'] = 0
    self.pointer['test'] = 0
    self.pointer['train'] = 0
    # DEBUG: OVERFIT. overfit.
    if debug == 'overfit':
      self.songs['train'] = self.songs['train'][0:1]
      #print (('DEBUG: trying to overfit on the following (repeating for train/validation/test):')
      for i in range(200):
        self.songs['train'].append(self.songs['train'][0])
      self.songs['validation'] = self.songs['train'][0:1]
      self.songs['test'] = self.songs['train'][0:1]
    #print (('lens: train: {}, val: {}, test: {}'.format(len(self.songs['train']), len(self.songs['validation']), len(self.songs['test'])))
    return self.songs

  def read_one_file(self, path, filename, pace_events):
    try:
      if debug:
        print (('Reading {}'.format(os.path.join(path,filename))))
      midi_pattern = midi.read_midifile(os.path.join(path,filename))
    except:
      print ( 'Error reading {}'.format(os.path.join(path,filename)))
      return None
    #
    # Interpreting the midi pattern.
    # A pattern has a list of tracks
    # (midi.Track()).
    # Each track is a list of events:
    #   * midi.events.SetTempoEvent: tick, data([int, int, int])
    #     (The three ints are really three bytes representing one integer.)
    #   * midi.events.TimeSignatureEvent: tick, data([int, int, int, int])
    #     (ignored)
    #   * midi.events.KeySignatureEvent: tick, data([int, int])
    #     (ignored)
    #   * midi.events.MarkerEvent: tick, text, data
    #   * midi.events.PortEvent: tick(int), data
    #   * midi.events.TrackNameEvent: tick(int), text(string), data([ints])
    #   * midi.events.ProgramChangeEvent: tick, channel, data
    #   * midi.events.ControlChangeEvent: tick, channel, data
    #   * midi.events.PitchWheelEvent: tick, data(two bytes, 14 bits)
    #
    #   * midi.events.NoteOnEvent:  tick(int), channel(int), data([int,int]))
    #     - data[0] is the note (0-127)
    #     - data[1] is the velocity.
    #     - if velocity is 0, this is equivalent of a midi.NoteOffEvent
    #   * midi.events.NoteOffEvent: tick(int), channel(int), data([int,int]))
    #
    #   * midi.events.EndOfTrackEvent: tick(int), data()
    #
    # Ticks are relative.
    #
    # Tempo are in microseconds/quarter note.
    #
    # This interpretation was done after reading
    # http://electronicmusic.wikia.com/wiki/Velocity
    # http://faydoc.tripod.com/formats/mid.htm
    # http://www.lastrayofhope.co.uk/2009/12/23/midi-delta-time-ticks-to-seconds/2/
    # and looking at some files. It will hopefully be enough
    # for the use in this project.
    #
    # We'll save the data intermediately with a dict representing each tone.
    # The dicts we put into a list. Times are microseconds.
    # Keys: 'freq', 'velocity', 'begin-tick', 'tick-length'
    #
    # 'Output ticks resolution' are fixed at a 32th note,
    #   - so 8 ticks per quarter note.
    #
    # This approach means that we do not currently support
    #   tempo change events.
    #
    # TODO 1: Figure out pitch.
    # TODO 2: Figure out different channels and instruments.
    #
    
    song_data = []
    
    # Tempo:
    ticks_per_quarter_note = float(midi_pattern.resolution)
    #print (('Resoluton: {}'.format(ticks_per_quarter_note))
    input_ticks_per_output_tick = ticks_per_quarter_note/self.output_ticks_per_quarter_note
    #if debug == 'overfit': input_ticks_per_output_tick = 1.0
    
    # Multiply with output_ticks_pr_input_tick for output ticks.
    for track in midi_pattern:
      last_event_input_tick=0
      not_closed_notes = []
      for event in track:
        if type(event) == midi.events.SetTempoEvent:
          pass # These are currently ignored
        elif (type(event) == midi.events.NoteOffEvent) or \
             (type(event) == midi.events.NoteOnEvent and \
              event.velocity == 0):
          retained_not_closed_notes = []
          for e in not_closed_notes:
            if tone_to_freq(event.data[0]) == e[FREQ]:
              event_abs_tick = float(event.tick+last_event_input_tick)/input_ticks_per_output_tick
              #current_note['length'] = float(ticks*microseconds_per_tick)
              e[LENGTH] = event_abs_tick-e[BEGIN_TICK]
              song_data.append(e)
            else:
              retained_not_closed_notes.append(e)
          #if len(not_closed_notes) == len(retained_not_closed_notes):
          #  print (('Warning. NoteOffEvent, but len(not_closed_notes)({}) == len(retained_not_closed_notes)({})'.format(len(not_closed_notes), len(retained_not_closed_notes)))
          #  print (('NoteOff: {}'.format(tone_to_freq(event.data[0])))
          #  print (('not closed: {}'.format(not_closed_notes))
          not_closed_notes = retained_not_closed_notes
        elif type(event) == midi.events.NoteOnEvent:
          begin_tick = float(event.tick+last_event_input_tick)/input_ticks_per_output_tick
          note = [0.0]*(NUM_FEATURES_PER_TONE+1)
          note[FREQ]       = tone_to_freq(event.data[0])
          note[VELOCITY]   = float(event.data[1])
          note[BEGIN_TICK] = begin_tick
          not_closed_notes.append(note)
          #not_closed_notes.append([0.0, tone_to_freq(event.data[0]), velocity, begin_tick, event.channel])
        last_event_input_tick += event.tick
      for e in not_closed_notes:
        #print (('Warning: found no NoteOffEvent for this note. Will close it. {}'.format(e))
        e[LENGTH] = float(ticks_per_quarter_note)/input_ticks_per_output_tick
        song_data.append(e)
    song_data.sort(key=lambda e: e[BEGIN_TICK])
    if (pace_events):
      pace_event_list = []
      pace_tick = 0.0
      song_tick_length = song_data[-1][BEGIN_TICK]+song_data[-1][LENGTH]
      while pace_tick < song_tick_length:
        song_data.append([0.0, 440.0, 0.0, pace_tick, 0.0])
        pace_tick += float(ticks_per_quarter_note)/input_ticks_per_output_tick
      song_data.sort(key=lambda e: e[BEGIN_TICK])
    return song_data

  def rewind(self, part='train'):
    self.pointer[part] = 0

  def get_batch(self, batchsize, songlength, part='train'):
    """
      get_batch() returns a batch from self.songs, as a
      pair of tensors (genrecomposer, song_data).
      
      The first tensor is a tensor of genres and composers
        (as two one-hot vectors that are concatenated).
      The second tensor contains song data.
        Song data has dimensions [batchsize, songlength, num_song_features]

      To have the sequence be the primary index is convention in
      tensorflow's rnn api.
      The tensors will have to be split later.
      Songs are currently chopped off after songlength.
      TODO: handle this in a better way.

      Since self.songs was shuffled in read_data(), the batch is
      a random selection without repetition.

      songlength is related to internal sample frequency.
      We fix this to be every 32th notes. # 50 milliseconds.
      This means 8 samples per quarter note.
      There is currently no notion of tempo in the representation.

      composer and genre is concatenated to each event
      in the sequence. There might be more clever ways
      of doing this. It's not reasonable to change composer
      or genre in the middle of a song.
      
      A tone  has a feature telling us the pause before it.

    """
    #print (('get_batch(): pointer: {}, len: {}, batchsize: {}'.format(self.pointer[part], len(self.songs[part]), batchsize))
    if self.pointer[part] > len(self.songs[part])-batchsize:
      return [None, None]
    if self.songs[part]:
      batch = self.songs[part][self.pointer[part]:self.pointer[part]+batchsize]
      self.pointer[part] += batchsize
      # subtract two for start-time and channel, which we don't include.
      num_meta_features = len(self.genres)+len(self.composers)
      # All features except timing are multiplied with tones_per_cell (default 1)
      num_song_features = NUM_FEATURES_PER_TONE*self.tones_per_cell+1
      batch_genrecomposer = np.ndarray(shape=[batchsize, num_meta_features])
      batch_songs = np.ndarray(shape=[batchsize, songlength, num_song_features])
      #print ( 'batch shape: {}'.format(batch_songs.shape)
      zeroframe = np.zeros(shape=[num_song_features])
      for s in range(len(batch)):
        songmatrix = np.ndarray(shape=[songlength, num_song_features])
        composeronehot = onehot(self.composers.index(batch[s][1]), len(self.composers))
        genreonehot = onehot(self.genres.index(batch[s][0]), len(self.genres))
        genrecomposer = np.concatenate([genreonehot, composeronehot])
        
        
        #random position:
        begin = 0
        if len(batch[s][SONG_DATA]) > songlength*self.tones_per_cell:
          begin = random.randint(0, len(batch[s][SONG_DATA])-songlength*self.tones_per_cell)
        matrixrow = 0
        n = begin
        while matrixrow < songlength:
          eventindex = 0
          event = np.zeros(shape=[num_song_features])
          if n < len(batch[s][SONG_DATA]):
            event[LENGTH]   = batch[s][SONG_DATA][n][LENGTH]
            event[FREQ]     = batch[s][SONG_DATA][n][FREQ]
            event[VELOCITY] = batch[s][SONG_DATA][n][VELOCITY]
            ticks_from_start_of_prev_tone = 0.0
            if n>0:
              # beginning of this tone, minus starting of previous
              ticks_from_start_of_prev_tone = batch[s][SONG_DATA][n][BEGIN_TICK]-batch[s][SONG_DATA][n-1][BEGIN_TICK]
              # we don't include start-time at index 0:
              # and not channel at -1.
            # tones are allowed to overlap. This is indicated with
            # relative time zero in the midi spec.
            event[TICKS_FROM_PREV_START] = ticks_from_start_of_prev_tone
            tone_count = 1
            for simultaneous in range(1,self.tones_per_cell):
              if n+simultaneous >= len(batch[s][SONG_DATA]):
                break
              if batch[s][SONG_DATA][n+simultaneous][BEGIN_TICK]-batch[s][SONG_DATA][n][BEGIN_TICK] == 0:
                offset = simultaneous*NUM_FEATURES_PER_TONE
                event[offset+LENGTH]   = batch[s][SONG_DATA][n+simultaneous][LENGTH]
                event[offset+FREQ]     = batch[s][SONG_DATA][n+simultaneous][FREQ]
                event[offset+VELOCITY] = batch[s][SONG_DATA][n+simultaneous][VELOCITY]
                tone_count += 1
              else:
                break
          songmatrix[matrixrow,:] = event
          matrixrow += 1
          n += tone_count
        #if s == 0 and self.pointer[part] == batchsize:
        #  print ( songmatrix[0:10,:]
        batch_genrecomposer[s,:] = genrecomposer
        batch_songs[s,:,:] = songmatrix
      #batched_sequence = np.split(batch_songs, indices_or_sections=songlength, axis=1)
      #return [np.squeeze(s, axis=1) for s in batched_sequence]
      #print (('batch returns [0:10]: {}'.format(batch_songs[0,0:10,:]))
      return [batch_genrecomposer, batch_songs]
    else:
      raise 'get_batch() called but self.songs is not initialized.'
  
  def get_num_song_features(self):
    return NUM_FEATURES_PER_TONE*self.tones_per_cell+1
  def get_num_meta_features(self):
    return len(self.genres)+len(self.composers)

  def get_midi_pattern(self, song_data):
    """
    get_midi_pattern takes a song in internal representation 
    (a tensor of dimensions [songlength, self.num_song_features]).
    the three values are length, frequency, velocity.
    if velocity of a frame is zero, no midi event will be
    triggered at that frame.

    returns the midi_pattern.

    Can be used with filename == None. Then nothing is saved, but only returned.
    """
    #print (('song_data[0:10]: {}'.format(song_data[0:10])))


    #
    # Interpreting the midi pattern.
    # A pattern has a list of tracks
    # (midi.Track()).
    # Each track is a list of events:
    #   * midi.events.SetTempoEvent: tick, data([int, int, int])
    #     (The three ints are really three bytes representing one integer.)
    #   * midi.events.TimeSignatureEvent: tick, data([int, int, int, int])
    #     (ignored)
    #   * midi.events.KeySignatureEvent: tick, data([int, int])
    #     (ignored)
    #   * midi.events.MarkerEvent: tick, text, data
    #   * midi.events.PortEvent: tick(int), data
    #   * midi.events.TrackNameEvent: tick(int), text(string), data([ints])
    #   * midi.events.ProgramChangeEvent: tick, channel, data
    #   * midi.events.ControlChangeEvent: tick, channel, data
    #   * midi.events.PitchWheelEvent: tick, data(two bytes, 14 bits)
    #
    #   * midi.events.NoteOnEvent:  tick(int), channel(int), data([int,int]))
    #     - data[0] is the note (0-127)
    #     - data[1] is the velocity.
    #     - if velocity is 0, this is equivalent of a midi.NoteOffEvent
    #   * midi.events.NoteOffEvent: tick(int), channel(int), data([int,int]))
    #
    #   * midi.events.EndOfTrackEvent: tick(int), data()
    #
    # Ticks are relative.
    #
    # Tempo are in microseconds/quarter note.
    #
    # This interpretation was done after reading
    # http://electronicmusic.wikia.com/wiki/Velocity
    # http://faydoc.tripod.com/formats/mid.htm
    # http://www.lastrayofhope.co.uk/2009/12/23/midi-delta-time-ticks-to-seconds/2/
    # and looking at some files. It will hopefully be enough
    # for the use in this project.
    #
    # This approach means that we do not currently support
    #   tempo change events.
    #
    
    # Tempo:
    # Multiply with output_ticks_pr_input_tick for output ticks.
    midi_pattern = midi.Pattern([], resolution=int(self.output_ticks_per_quarter_note))
    cur_track = midi.Track([])
    cur_track.append(midi.events.SetTempoEvent(tick=0, bpm=45))
    future_events = {}
    last_event_tick = 0
    
    ticks_to_this_tone = 0.0
    song_events_absolute_ticks = []
    abs_tick_note_beginning = 0.0
    for frame in song_data:
      abs_tick_note_beginning += frame[TICKS_FROM_PREV_START]
      for subframe in range(self.tones_per_cell):
        offset = subframe*NUM_FEATURES_PER_TONE
        tick_len           = int(round(frame[offset+LENGTH]))
        freq               = frame[offset+FREQ]
        velocity           = min(int(round(frame[offset+VELOCITY])),127)
        #print (('tick_len: {}, freq: {}, velocity: {}, ticks_from_prev_start: {}'.format(tick_len, freq, velocity, frame[TICKS_FROM_PREV_START]))
        d = freq_to_tone(freq)
        #print (('d: {}'.format(d))
        if d is not None and velocity > 0 and tick_len > 0:
          # range-check with preserved tone, changed one octave:
          tone = d['tone']
          while tone < 0:   tone += 12
          while tone > 127: tone -= 12
          pitch_wheel = cents_to_pitchwheel_units(d['cents'])
          #print (('tick_len: {}, freq: {}, tone: {}, pitch_wheel: {}, velocity: {}'.format(tick_len, freq, tone, pitch_wheel, velocity))
          #if pitch_wheel != 0:
          #midi.events.PitchWheelEvent(tick=int(ticks_to_this_tone),
          #                                            pitch=pitch_wheel)
          song_events_absolute_ticks.append((abs_tick_note_beginning,
                                             midi.events.NoteOnEvent(
                                                   tick=0,
                                                   velocity=velocity,
                                                   pitch=tone)))
          song_events_absolute_ticks.append((abs_tick_note_beginning+tick_len,
                                             midi.events.NoteOffEvent(
                                                    tick=0,
                                                    velocity=0,
                                                    pitch=tone)))
    song_events_absolute_ticks.sort(key=lambda e: e[0])
    abs_tick_note_beginning = 0.0
    for abs_tick,event in song_events_absolute_ticks:
      rel_tick = abs_tick-abs_tick_note_beginning
      event.tick = int(round(rel_tick))
      cur_track.append(event)
      abs_tick_note_beginning=abs_tick
    
    cur_track.append(midi.EndOfTrackEvent(tick=int(self.output_ticks_per_quarter_note)))
    midi_pattern.append(cur_track)
    #print ( 'print (ing midi track.'
    #print ( midi_pattern
    return midi_pattern

  def save_midi_pattern(self, filename, midi_pattern):
    if filename is not None:
      midi.write_midifile(filename, midi_pattern)

  def save_data(self, filename, song_data):
    """
    save_data takes a filename and a song in internal representation 
    (a tensor of dimensions [songlength, 3]).
    the three values are length, frequency, velocity.
    if velocity of a frame is zero, no midi event will be
    triggered at that frame.

    returns the midi_pattern.

    Can be used with filename == None. Then nothing is saved, but only returned.
    """
    midi_pattern = self.get_midi_pattern(song_data)
    self.save_midi_pattern(filename, midi_pattern)
    return midi_pattern

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
  if freq <= 0.0:
    return None
  float_tone = (69.0+12*math.log(float(freq)/440.0, 2))
  int_tone = int(float_tone)
  cents = int(1200*math.log(float(freq)/tone_to_freq(int_tone), 2))
  return {'tone': int_tone, 'cents': cents}

def cents_to_pitchwheel_units(cents):
  return int(40.96*(float(cents)))

def onehot(i, length):
  a = np.zeros(shape=[length])
  a[i] = 1
  return a



def main():
  filename = sys.argv[1]
  print (('File: {}'.format(filename)))
  dl = MusicDataLoader(datadir=None, select_validation_percentage=0.0, select_test_percentage=0.0)
  print (('length, frequency, velocity, time from previous start.'))
  abs_song_data = dl.read_one_file(os.path.dirname(filename), os.path.basename(filename), pace_events=True)
  
  rel_song_data = []
  last_start = None
  for i,e in enumerate(abs_song_data):
    this_start = e[3]
    if last_start:
      e[3] = e[3]-last_start
    rel_song_data.append(e)
    last_start = this_start
    print ((e))
  if len(sys.argv) > 2:
    if not os.path.exists(sys.argv[2]):
      print (('Saving: {}.'.format(sys.argv[2])))
      dl.save_data(sys.argv[2], rel_song_data)
    else:
      print (('File already exists: {}. Not saving.'.format(sys.argv[2])))
if __name__ == "__main__":
  main()


