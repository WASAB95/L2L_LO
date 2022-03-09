;; SpikingLab: Modelling Spiking Neural Networks with STDP learning

;; Author: Cristian Jimenez Romero - The Open University - 2015
;; Publication: SpikingLab: modelling agents controlled by Spiking Neural Networks in Netlogo
;; C Jimenez-Romero, J Johnson
;; Neural Computing and Applications 28 (1), 755-764

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;SpikingLab SNN related code:;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
breed [neurontypes neurontype]
breed [inputneurons inputneuron]
breed [normalneurons normalneuron]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Demo: Artificial insect related code:;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
breed [testcreatures testcreature]
breed [visualsensors visualsensor]
breed [pheromonesensors pheromonesensor]
breed [nestsensors nestsensor]

;breed [sightlines sightline]
directed-link-breed [sight-trajectories sight-trajectory]

directed-link-breed [normal-synapses normal-synapse]
directed-link-breed [input-synapses input-synapse]

patches-own [
  chemical             ;; amount of chemical on this patch
  food                 ;; amount of food on this patch (0, 1, or 2)
  nest?                ;; true on nest patches, false elsewhere
  nest-scent           ;; number that is higher closer to the nest
  food-source-number   ;; number (1, 2, or 3) to identify the food sources
  visual_object       ;; type (0 = empty, 1= wall, 2= obstacle, 3=food)
]

;;;
;;; Global variables including SpikingLab internals
;;;
globals [
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;SNN Module globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  neuron_state_open;; describe state machine of the neuron
  neuron_state_firing;; describe state machine of the neuron
  neuron_state_refractory;; describe state machine of the neuron
  excitatory_synapse
  inhibitory_synapse
  system_iter_unit ;;time steps of each iteration expressed in milliseconds
  plot-list
  plot-list2
  PulseHistoryBuffSize ;;Size of pulse history buffer
  input_value_empty ;;Indicate that there is no input value
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Demo globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  pspikefrequency ;;Number of spikes emitted by an input neuron in response to one stimulus
  error_free_counter ;;Number of iterations since the insect collided with a noxious stimulus
  required_error_free_iterations ;;Number of error-free iterations necessary to stop training
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;World globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  min_x_boundary ;;Define initial x coordinate of simulation area
  min_y_boundary ;;Define initial y coordinate of simulation area
  max_x_boundary ;;Define final x coordinate of simulation area
  max_y_boundary ;;Define final y coordinate of simulation area

  nestx
  nesty
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;GA Globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  fitness_value
  simulation_end_signal
  ;;;;;;;;;;;;;;;;;;;;;;;;;;Ants Globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  simulation_area

  raw_weight_data
  raw_plasticity_data
  raw_delay_data

  weight_data
  delay_data
  plasticity_data

]

neurontypes-own [

  neurontypeid
  ;;;;;;;;;;Neuron Dynamics;;;;;;;;;;;
  restingpotential
  firethreshold
  decayrate ;Decay rate constant
  relrefractorypotential ;Refractory potential
  intervrefractoryperiod ;Duration of absolute-refractory period
  minmembranepotential ;lowest boundary for membrane potential
  ;;;;;;;Learning Parameters;;;;;;;;;
  pos_hebb_weight ;;Weight to increase the efficacy of synapses
  pos_time_window ;; Positive learning window
  neg_hebb_weight ;;Weight to decrease the efficacy of synapses
  neg_time_window ;; negative learning window
  max_synaptic_weight ;;Maximum synaptic weight
  min_synaptic_weight ;;Minimum synaptic weight
  pos_syn_change_interval ;;ranges of pre–to–post synaptic interspike intervals over which synaptic change occur.
  neg_syn_change_interval

]


normalneurons-own [

  nlayernum
  nneuronid
  nneuronstate
  nrestingpotential
  nfirethreshold
  nmembranepotential
  ndecayrate
  nrelrefractorypotential
  nintervrefractoryperiod
  nrefractorycounter
  naxondelay
  nsynapsesarray
  nnumofsynapses
  nlast-firing-time
  nincomingspikes
  nlastspikeinput
  nneuronlabel
  nminmembranepotential
  ;;;;;;;;;;;;;;;;Learning Parameters;;;;;;;;;;;;;;;;;
  npos_hebb_weight ;;Weight to increase the efficacy of synapses
  npos_time_window ;; Positive learning window
  nneg_hebb_weight ;;Weight to decrease the efficacy of synapses
  nneg_time_window ;; negative learning window
  nmax_synaptic_weight ;;Maximum synaptic weight
  nmin_synaptic_weight ;;Minimum synaptic weight
  npos_syn_change_interval ;;ranges of pre–to–post synaptic interspike intervals over which synaptic change occur.
  nneg_syn_change_interval

]


normal-synapses-own [

  presynneuronlabel
  possynneuronlabel
  presynneuronid
  possynneuronid
  synapseefficacy
  exc_or_inh
  synapsedelay
  joblist
  learningon?

]


inputneurons-own [

   layernum
   neuronid
   neuronstate
   pulsecounter ;;Count the number of sent pulses
   interspikecounter ;;Count the time between pulses
   numberofspikes ;;Number of spikes to send
   postsynneuron
   encodedvalue
   isynapseefficacy ;;In most cases should be large enough to activate possynn with a single spike
   neuronlabel

]


testcreatures-own [

  creature_label
  creature_id
  reward_neuron
  pain_neuron
  move_neuron
  rotate_left_neuron
  rotate_right_neuron
  pheromone_neuron
  nest_neuron
  creature_sightline
  last_nest_collision
  last_food_collision
  carrying_food_amount
  fitness

]

pheromonesensors-own [
  sensor_id
  relative_rotation
  left_sensor_attached_to_neuron
  middle_sensor_attached_to_neuron
  right_sensor_attached_to_neuron
  attached_to_creature
]

nestsensors-own [
  sensor_id
  relative_rotation
  left_sensor_attached_to_neuron
  middle_sensor_attached_to_neuron
  right_sensor_attached_to_neuron
  attached_to_creature
]

visualsensors-own [

  sensor_id
  perceived_stimuli
  distance_to_stimuli
  relative_rotation ;;Position relative to front
  attached_to_colour
  attached_to_neuron
  attached_to_creature

]

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Patches procedures:
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to setup-patches
  ask patches
  [ setup-nest
    setup-food
    recolor-patch ]
end

to setup-nest  ;; patch procedure
  if visual_object != 0 [ set nest? false stop ]
  ;; set nest? variable to true inside the nest, false elsewhere
  set nest? (distancexy nestx nesty) < nest_size ;82 30
  ;; spread a nest-scent over the whole world -- stronger near the nest
  set nest-scent 200 - distancexy nestx nesty
end

to setup-food  ;; patch procedure
  if visual_object = 1 [ stop ]
  ;; setup food source one on the right
  if (distancexy (0.7 * max-pxcor) 53) < food_source_size
  [ set food-source-number 1 set visual_object 3 ]
  ;; setup food source two on the lower-left
  if (distancexy (0.7 * max-pxcor) 6) < food_source_size
  [ set food-source-number 2 set visual_object 3 ]
  ;; setup food source three on the upper-left
  if (distancexy (0.96 * max-pxcor) 30) < food_source_size
  [ set food-source-number 3 set visual_object 3 ]
  ;; set "food" at sources to either 1 or 2, randomly
  if food-source-number > 0
  [ set food 2 ] ;one-of [2 4] ]

end

to recolor-patch  ;; patch procedure
  if visual_object = 1 [ stop ]
  ;; give color to nest and food sources
  ifelse nest?
  [ set pcolor violet ]
  [ ifelse food > 0
    [
      set pcolor green
      ;if food-source-number = 1 [ set pcolor green ] ;cyan
      ;if food-source-number = 2 [ set pcolor green  ] ;sky
      ;if food-source-number = 3 [ set pcolor green ]
    ] ;blue
    ;; scale color to show chemical concentration
    [ set pcolor scale-color blue chemical 0.1 5 ]
  ]
end

to update-patches
  ask patches
  [ set chemical chemical * (100 - evaporation-rate) / 100  ;; slowly evaporate chemical
    recolor-patch ]
end

;;;
;;; Set global variables with their initial values
;;;
to initialize-global-vars

  set system_iter_unit 1 ;; each simulation iteration represents 1 time unit
  set neuron_state_open 1 ;;State machine idle
  set neuron_state_firing 2 ;;State machine firing
  set neuron_state_refractory 3 ;;State machine refractory
  set excitatory_synapse 1
  set inhibitory_synapse 2
  set plot-list[] ;;List for spike history
  set plot-list2[] ;;List for spike history
  set PulseHistoryBuffSize 30
  set input_value_empty -1

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;Simulation area globals;;;;;;;;;;;;;;;;;;;;;;;;
  set min_x_boundary 60  ;;Define initial x coordinate of simulation area
  set min_y_boundary 1   ;;Define initial y coordinate of simulation area
  set max_x_boundary 100 ;;Define final x coordinate of simulation area
  set max_y_boundary 60  ;;Define final y coordinate of simulation area

  set nestx 72 + random(14)
  set nesty 18 + random(18)
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Insect globals;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  set pspikefrequency 1
  set error_free_counter 0
  set required_error_free_iterations 35000

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;Genetic Algorithm ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  set fitness_value 0

  set weight_data[]
  set delay_data[]
  set plasticity_data[]

  set simulation_end_signal false

end

;;;
;;; Create a neuron type
;;; Parameters: Type-identifier, resting potential, firing threshold, decay rate, refractory potential, duration of refractory period, lowest boundary for membrane potential
to setup-neurontype [#pneurontypeid #prestpot #pthreshold #pdecayr #prefractpot #pintrefrectp #minmembranepotential]

  create-neurontypes 1
  [
     set shape "square"
     set neurontypeid #pneurontypeid
     set restingpotential #prestpot
     set firethreshold #pthreshold
     set decayrate #pdecayr
     set relrefractorypotential #prefractpot
     set intervrefractoryperiod #pintrefrectp
     set minmembranepotential #minmembranepotential
     ht
  ]

end


;;;
;;; Set learning parameters for neuron type pneurontypeid
;;;

to set-neurontype-learning-params [ #pneurontypeid #ppos_hebb_weight #ppos_time_window #pneg_hebb_weight #pneg_time_window #pmax_synaptic_weight
                                    #pmin_synaptic_weight #ppos_syn_change_interval #pneg_syn_change_interval]

  ask neurontypes with [neurontypeid = #pneurontypeid]
  [
    set pos_hebb_weight #ppos_hebb_weight
    set pos_time_window #ppos_time_window
    set neg_hebb_weight #pneg_hebb_weight
    set neg_time_window #pneg_time_window
    set max_synaptic_weight #pmax_synaptic_weight
    set min_synaptic_weight #pmin_synaptic_weight
    set pos_syn_change_interval #ppos_syn_change_interval
    set neg_syn_change_interval #pneg_syn_change_interval
  ]

end

;;;
;;; Declare an existing neuron "pneuronid" as neuron-type "pneurontypeid"
;;;
to set-neuron-to-neurontype [ #pneurontypeid #pneuronid ] ;Call by observer


    ask neurontypes with [neurontypeid = #pneurontypeid]
    [
      ask normalneuron #pneuronid
      [
        set nrestingpotential [restingpotential] of myself
        set nmembranepotential [restingpotential] of myself
        set nfirethreshold [firethreshold] of myself
        set ndecayrate [decayrate] of myself
        set nrelrefractorypotential [relrefractorypotential] of myself
        set nintervrefractoryperiod [intervrefractoryperiod] of myself
        set nminmembranepotential [minmembranepotential] of myself

        ;;;;;;;;;;;;;;;;Learning Parameters;;;;;;;;;;;;;;;;;
        set npos_hebb_weight [pos_hebb_weight] of myself
        set npos_time_window [pos_time_window] of myself
        set nneg_hebb_weight [neg_hebb_weight] of myself
        set nneg_time_window [neg_time_window] of myself
        set nmax_synaptic_weight [max_synaptic_weight] of myself
        set nmin_synaptic_weight [min_synaptic_weight] of myself
        set npos_syn_change_interval [pos_syn_change_interval] of myself
        set nneg_syn_change_interval [neg_syn_change_interval] of myself
      ]
     ]

end


;;;
;;; Get intern identifier of the input-neuron with label = #pneuronlabel
;;;
to-report get-input-neuronid-from-label [#pneuronlabel]

  let returned_id nobody
  ask one-of inputneurons with [neuronlabel = #pneuronlabel][set returned_id neuronid]
  report returned_id

end

;;;
;;; Get intern identifier of the normal-neuron with label = #pneuronlabel
;;;
to-report get-neuronid-from-label [#pneuronlabel]

  let returned_id nobody
  ask one-of normalneurons with [nneuronlabel = #pneuronlabel][set returned_id nneuronid]
  report returned_id

end


;;;
;;; Create a new synapse between pre-synaptic neuron: #ppresynneuronlabel and post-synaptic neuron: #ppossynneuronlabel
;;;
to setup-synapse [#ppresynneuronlabel #ppossynneuronlabel #psynapseefficacy #pexc_or_inh #psyndelay #plearningon?]

   let presynneuid get-neuronid-from-label #ppresynneuronlabel
   let possynneuid get-neuronid-from-label #ppossynneuronlabel

   let postsynneu normalneuron possynneuid
   ask normalneuron presynneuid [
        create-normal-synapse-to postsynneu [
           set presynneuronlabel #ppresynneuronlabel
           set possynneuronlabel #ppossynneuronlabel
           set presynneuronid presynneuid
           set possynneuronid possynneuid
           set synapseefficacy #psynapseefficacy
           set exc_or_inh #pexc_or_inh
           set synapsedelay #psyndelay
           set joblist []
           set learningon? #plearningon?
           ifelse (#pexc_or_inh = inhibitory_synapse)
           [
             set color red
           ]
           [
             set color grey
           ]
        ]
     ]

end


;;;
;;; Process incoming pulse from input neuron
;;;
to receive-input-neuron-pulse [ #psnefficacy #pexcinh ];;called by Neuron

  if ( nneuronstate != neuron_state_refractory )
  [
      ;;Adjust membrane potential:
      ifelse ( #pexcinh = excitatory_synapse )
      [
          set nmembranepotential nmembranepotential + #psnefficacy  ;;increase membrane potential
      ]
      [
          set nmembranepotential nmembranepotential - #psnefficacy ;;decrease membrane potential
          if (nmembranepotential < nminmembranepotential) ;; Floor for the membrane potential in case of extreme inhibition
          [
             set nmembranepotential nminmembranepotential
          ]
      ]
  ]
  set nlastspikeinput ticks

end


;;;
;;; Neuron pstneuronid# processes incoming pulse from neuron #prneuronid
;;;
to receive-pulse [ #prneuronid #snefficacy #excinh #plearningon? ];;called by neuron

  if ( nneuronstate != neuron_state_refractory )
  [
      ;;Perturb membrane potential:
      ifelse ( #excinh = excitatory_synapse )
      [
          set nmembranepotential nmembranepotential + (#snefficacy * epsp-kernel) ;;increase membrane potential
      ]
      [
          set nmembranepotential nmembranepotential - (#snefficacy * ipsp-kernel) ;;decrease membrane potential
          if (nmembranepotential < nminmembranepotential) ;; Floor for the membrane potential in case of extreme inhibition
          [
             set nmembranepotential nminmembranepotential
          ]
      ]
  ]
  ;;Remember last input spike:
  set nlastspikeinput ticks
  ;; If plasticity is activated then store pulse info for further processing and apply STDP
  if (#plearningon? and istrainingmode?)
  [
     ;;Don't let incoming-pulses history-buffer grow beyond limits (delete oldest spike):
     check-pulse-history-buffer
     ;;Create list of parameters and populate it:
     let pulseinflist[]
     set pulseinflist lput #prneuronid pulseinflist ;;Add Presynaptic neuron Id
     set pulseinflist lput #snefficacy pulseinflist ;;Add Synaptic efficacy
     set pulseinflist lput #excinh pulseinflist ;;Add excitatory or inhibitory info.
     set pulseinflist lput ticks pulseinflist ;;Add arriving time
     set pulseinflist lput false pulseinflist ;;Indicate if pulse has been processed as an EPSP following a Postsynaptic spike ( Post -> Pre, negative hebbian)
     ;;Add list of parameters to list of incoming pulses:
     set nincomingspikes lput pulseinflist nincomingspikes
     ;;Apply STDP learning rule:
     apply-stdp-learning-rule
  ]

end


;;;
;;; Neuron prepares to fire
;;;
to prepare-pulse-firing ;;Called by Neurons

   ;;Remember last firing time
   set nlast-firing-time ticks
   ;;Apply learning rule and after that empty incoming-pulses history:
   apply-stdp-learning-rule
   empty-pulse-history-buffer
   ;;transmit Pulse to postsynaptic neurons:
   propagate-pulses
   ;;Set State to refractory
   set nneuronstate neuron_state_refractory
   ;;initialize counter of refractory period in number of iterations
   set nrefractorycounter nintervrefractoryperiod

end


;;;
;;; Kernel for inhibitory post-synaptic potential
;;;
to-report ipsp-kernel ;;Called by Neurons

  report 1

end


;;;
;;; Kernel for excitatory post-synaptic potential
;;;
to-report epsp-kernel ;;Called by Neurons

  report 1

end


;;;
;;; Kernel for membrane decay towards resting potential (If current membrane pot. > Resting pot.)
;;;
to-report negative-decay-kernel ;;Called by Neurons

  report  abs(nrestingpotential - nmembranepotential) * ndecayrate

end


;;;
;;; Kernel for membrane decay towards resting potential (If current membrane pot. < Resting pot.)
;;;
to-report positive-decay-kernel ;;Called by Neurons

  report abs(nrestingpotential - nmembranepotential) * ndecayrate

end


;;;
;;; Bring membrane potential towards its resting state
;;;
to decay ;;Called by Neurons
    ;;Move membrane potential towards resting potential:
    ifelse (nmembranepotential > nrestingpotential )
    [
        let expdecay negative-decay-kernel ;
        ifelse ((nmembranepotential - expdecay) < nrestingpotential)
        [
          set nmembranepotential nrestingpotential
        ]
        [
          set nmembranepotential nmembranepotential - expdecay
        ]
    ]
    [
      let expdecay positive-decay-kernel ;
      ifelse ((nmembranepotential + expdecay) > nrestingpotential)
      [
        set nmembranepotential nrestingpotential
      ]
      [
        set nmembranepotential nmembranepotential + expdecay
      ]
    ]

end


;;;
;;; Process neuron dynamics according to its machine state
;;;
to do-neuron-dynamics ;;Called by Neurons

  ifelse ( nneuronstate = neuron_state_open )
  [
    if (nmembranepotential != nrestingpotential )
    [
      ;;Check if membrane potential reached the firing threshold
      ifelse( nmembranepotential >= nfirethreshold )
      [
        prepare-pulse-firing
        set color red
      ]
      [
         ;;Move membrane potential towards resting potential:
         decay
      ]
    ]
  ]
  [
    ;;Not idle and not firing, then refractory:
    set color pink ;;Restore normal colour
    ;;Decrease timer of absolute refractory period:
    set nrefractorycounter nrefractorycounter - system_iter_unit
    ;;Set membrane potential with refractory potential:
    set nmembranepotential nrelrefractorypotential
    if ( nrefractorycounter <= 0) ;;End of absolute refractory period?
    [
      ;;Set neuron in open state:
      set nneuronstate neuron_state_open
    ]
  ]
  ;;Continue with Axonal dynamics independently of the neuron state:
  do-synaptic-dynamics

end


;;;
;;; Delete history of incoming spikes
;;;
to empty-pulse-history-buffer ;;Called by neurons

  set nincomingspikes[]

end


;;;
;;; Apply the Spike Timing Dependent Plasticity rule
;;;
to apply-stdp-learning-rule ;;Call by neurons

  ;Apply rule: Ap.exp(dt/Tp); if dt < 0; dt = prespt - postspt
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  let currspikeinfo[]
  let itemcount 0
  let deltaweight 0

  while [itemcount < ( length nincomingspikes ) ]
  [
     set currspikeinfo ( item itemcount nincomingspikes ) ;;Get spike info:   prneuronid[0], snefficacy[1], excinh[2], arrivaltime[3], processedBynegHebb[4]

     ifelse ( item 2 currspikeinfo ) = excitatory_synapse ;;Is the spike coming from an excitatory synapsis?
     [
       let deltaspike ( item 3 currspikeinfo ) - nlast-firing-time
       if ( deltaspike >= nneg_time_window and deltaspike <= npos_time_window) ;;Is spike within learning window?
       [
           ;;Calculate learning factor:
          ifelse ( deltaspike <= 0 ) ;;Increase weight
          [
            set deltaweight  npos_hebb_weight * exp(deltaspike / npos_syn_change_interval )
            ask normal-synapse  ( item 0 currspikeinfo ) nneuronid  [update-synapse-efficacy deltaweight [nmax_synaptic_weight] of myself [nmin_synaptic_weight] of myself]
          ]
          [
            if (( item 4 currspikeinfo ) = false) ;;if spike has not been processed then compute Hebb rule:
            [
              set deltaweight (- nneg_hebb_weight * exp(- deltaspike / nneg_syn_change_interval )) ;;Turn positive delta into a negative weight
              set currspikeinfo replace-item 4 currspikeinfo true ;Indicate that this pulse has already been processed as a EPSP after Postsyn neuron has fired (negative hebbian)
              set nincomingspikes replace-item itemcount nincomingspikes currspikeinfo
              ask normal-synapse  ( item 0 currspikeinfo ) nneuronid [update-synapse-efficacy deltaweight [nmax_synaptic_weight] of myself [nmin_synaptic_weight] of myself]
            ]
          ]
       ]
     ]
     [
       ;;Inhibitory Synapses: Plasticity in inhibitory synapses not implemented yet

     ]
     set itemcount itemcount + 1
  ]

end


;;;
;;; Don't store more pulses than the specified by PulseHistoryBuffSize
;;;
to check-pulse-history-buffer ;;Call by neurons

  if( length nincomingspikes > PulseHistoryBuffSize )
  [
    ;; Remove oldest pulse in the list
    set nincomingspikes remove-item 0 nincomingspikes
  ]

end


;;;
;;; Propagate pulse to all post-synaptic neurons
;;;
to propagate-pulses ;;Call by neurons

  ;; Insert a new pulse in all synapses having the current neuron as presynaptic
  ask my-out-normal-synapses
  [
    add-pulse-job
  ]

end


;;;
;;; Process synaptic dynamics of synapses with pre-synaptic neuron: presynneuronid
;;;
to do-synaptic-dynamics ;;Call by neurons

  ;; Process all outgoing synapses with pulses in their job-list
  ask my-out-normal-synapses with [ length joblist > 0 ]
  [
    process-pulses-queue
  ]

end


;;;
;;; Enqueue pulse in synapse
;;;
to add-pulse-job ;;Call by link (synapse)

    ;;Add a new Pulse with its delay time at the end of the outgoing-synapse joblist
    set joblist lput synapsedelay joblist

end


;;;
;;; Change synaptic weight
;;;
to update-synapse-efficacy [ #deltaweight #pmax_synaptic_weight #pmin_synaptic_weight] ;;Call by synapse

  ifelse ( synapseefficacy + #deltaweight ) > #pmax_synaptic_weight
  [
    set synapseefficacy #pmax_synaptic_weight
  ]
  [
    ifelse ( synapseefficacy + #deltaweight ) < #pmin_synaptic_weight
    [
      set synapseefficacy #pmin_synaptic_weight
    ]
    [
      set synapseefficacy synapseefficacy + #deltaweight
    ]
  ]

end

;;;
;;; For each traveling pulse in synapse check if pulse has already arrived at the post-synaptic neuron
;;;
to process-pulses-queue ;;Call by synapse

  set joblist map [ i -> i - 1  ] joblist ;;Decrease all delay counters by 1 time-unit

  foreach filter [i -> i <= 0] joblist ;;Process pulses that reached the postsynaptic neuron
  [
       ;;Transmit Pulse to Postsyn Neuron:
       ask other-end [receive-pulse [presynneuronid] of myself [synapseefficacy] of myself [exc_or_inh] of myself [learningon?] of myself]
  ]
  ;;Keep only "traveling" pulses in the list :
  ;;set joblist filter [? > 0] joblist
  set joblist filter [i -> i > 0] joblist

end


;;;
;;; Create one input neuron and attach it to neuron with label #ppostsynneuronlabel (input neurons have one connection only)
;;;
to setup-input-neuron [#pposx #pposy #label #ppostsynneuronlabel #psynapseefficacy #pcoding #pnumofspikes]

  let postsynneuronid get-neuronid-from-label #ppostsynneuronlabel
  set-default-shape inputneurons "square"
  create-inputneurons 1
  [
     set layernum 0
     set neuronid who
     set neuronstate neuron_state_open
     set pulsecounter 0
     set interspikecounter 0
     set numberofspikes #pnumofspikes
     set postsynneuron postsynneuronid
     set encodedvalue input_value_empty
     set isynapseefficacy #psynapseefficacy
     setxy #pposx #pposy
     set color green
     set label #label
     set neuronlabel #label
     setup-input-synapse
  ]

end


;;;
;;; Process pulses in input neuron
;;;
to do-input-neuron-dynamics ;;Called by inputneurons

  if ( pulsecounter > 0 ) ;;process only if input-neuron has something to do
  [
    set interspikecounter interspikecounter + 1
    if (interspikecounter > encodedvalue)
    [
    ;;Transmit pulse to Post-synaptic Neuron;
      ask out-input-synapse-neighbors [receive-input-neuron-pulse [isynapseefficacy] of myself [excitatory_synapse] of myself]
      set interspikecounter 0
      set pulsecounter pulsecounter - 1
    ]
  ]

end


;;;
;;; Encode input value (integer number) into pulses
;;;
to-report set-input-value [#pencodedvalue] ;;Called by inputneurons

  ;;Check if input neuron is ready to receive input
  let inputready false
  if ( pulsecounter = 0 )
  [
    set encodedvalue #pencodedvalue
    set pulsecounter numberofspikes ;;Initialize counter with the number of pulses to transmit with the encoded value
    set interspikecounter 0
    set inputready true
  ]
  report inputready
end


;;;
;;; Ask input neuron with id = #pneuronid to accept and encode a new input value
;;;
to feed-input-neuron [#pneuronid #pencodedvalue];;Called by observer

   ask inputneuron #pneuronid
   [
     let inputready set-input-value #pencodedvalue
   ]

end


;;;
;;; Ask input neuron with label = #pneuronlabel to accept and encode a new input value
;;;
to feed-input-neuron_by_label [#pneuronlabel #pencodedvalue];;Called by observer

   ask one-of inputneurons with [ neuronlabel = #pneuronlabel ]
   [
     let inputready set-input-value #pencodedvalue
   ]

end


;;;
;;; Create link to represent synapse from input neuron to post-synaptic neuron: postsynneuron
;;;
to setup-input-synapse ;;Call from inputneurons

   let psnneuron postsynneuron
   let postsynneu one-of (normalneurons with [nneuronid = psnneuron])
   create-input-synapse-to postsynneu

end


;;;
;;; Create and initialize neuron
;;;
to setup-normal-neuron [#playernum #pposx #pposy #label #pneurontypeid]

  set-default-shape normalneurons "circle"
  let returned_id nobody
  create-normalneurons 1
  [
     set nlayernum #playernum
     set nneuronid who
     set nneuronstate neuron_state_open
     set nrefractorycounter 0
     set nincomingspikes[]
     set nnumofsynapses 0
     set nlastspikeinput 0
     setxy #pposx #pposy
     set color pink
     set label #label
     set nneuronlabel #label
     set returned_id nneuronid
  ]
  set-neuron-to-neurontype #pneurontypeid returned_id

end


;;;
;;; Process neural dynamics
;;;
to do-network-dynamics

 ask inputneurons [ do-input-neuron-dynamics ] ;Process input neurons in random order
 ask normalneurons [do-neuron-dynamics] ;Process normal neurons in random order

end


;;;
;;; Show information about neuron with id: #srcneuron
;;;
to show-neuron-info [#srcneuron] ;;Called by observer

   ;set plot-list lput ( [nmembranepotential] of one-of normalneurons with [nneuronid = neuronid_monitor1] ) plot-list
   ask normalneurons with [nneuronid = #srcneuron]
   [
      print ""
      write "Neuron Id:" write nneuronid print ""
      write "Layer " write nlayernum print ""
      write "Membrane potential: " write nmembranepotential print ""
      write "Spike threshold: " write nfirethreshold print ""
      write "Resting potential: " write nrestingpotential print ""
      write "Refractory potential: " write nrelrefractorypotential print ""
      write "Last input-pulse received at:" write nlastspikeinput print ""
      write "Last spike fired at: " write nlast-firing-time print ""
      write "Current state: " write (ifelse-value ( nneuronstate = neuron_state_open ) ["idle"] ["refractory"] ) print ""
      print ""
   ]

end


;;;
;;; Show information about synapse with pre-synaptic neuron: #srcneuron and post-synaptic neuron: #dstneuron
;;;
to show-synapse-info-from-to [#srcneuron #dstneuron] ;;Called by observer

   ask normal-synapses with [presynneuronid = #srcneuron and possynneuronid = #dstneuron]
   [
      print "Synapse from:"
      write "Neuron " write #srcneuron write " to neuron " write #dstneuron print ""
      write "Weight: " write synapseefficacy print ""
      write "Delay: " write synapsedelay write "iteration(s)" print ""
      write "Excitatory or Inhibitory: "
      write (ifelse-value ( exc_or_inh = excitatory_synapse ) ["Excitatory"] ["Inhibitory"] )
      print ""
   ]

end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;End of SNN code;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;
;;; Create insect agent
;;;
to-report create-creature [#xpos #ypos #creature_label #reward_neuron_label #pain_neuron_label #move_neuron_label #rotate_left_neuron_label #rotate_right_neuron_label #pheromone_neuron_label #on_nest_input_neuron ]

  let reward_neuron_id get-input-neuronid-from-label #reward_neuron_label
  let pain_neuron_id get-input-neuronid-from-label #pain_neuron_label
  let move_neuron_id get-neuronid-from-label #move_neuron_label
  let rotate_left_neuron_id get-neuronid-from-label #rotate_left_neuron_label
  let rotate_right_neuron_id get-neuronid-from-label #rotate_right_neuron_label
  let pheromone_neuron_id get-neuronid-from-label #pheromone_neuron_label
  let on_nest_input_neuron_id get-input-neuronid-from-label #on_nest_input_neuron
  let returned_id nobody
  create-testcreatures 1 [
    set shape "bug"
    setxy #xpos #ypos
    set size 2.0
    set color yellow
    set creature_label #creature_label
    set reward_neuron reward_neuron_id
    set pain_neuron pain_neuron_id
    set move_neuron move_neuron_id
    set rotate_left_neuron rotate_left_neuron_id
    set rotate_right_neuron rotate_right_neuron_id
    set pheromone_neuron pheromone_neuron_id
    set nest_neuron on_nest_input_neuron_id
    set creature_id who
    set returned_id creature_id
    set last_nest_collision 0
    set last_food_collision 0
    set carrying_food_amount 0
    set fitness 0
  ]
  report  returned_id

end

to check-creature-on-nest-or-food
  let food_amount 0
  let on_nest? false
  ask patch-here [ set food_amount food set on_nest? nest?]
  ifelse on_nest? [
    set last_nest_collision ticks
    set carrying_food_amount 0
    feed-input-neuron nest_neuron 1
  ]
  [
    if food_amount > 0 and carrying_food_amount < 1 [
      update-fitness 1.5
      set last_food_collision ticks
      set food food - 1
      ifelse bring_food_to_nest? [
        set carrying_food_amount carrying_food_amount + 1 ;; Don't eat food and carry it to the nest
      ]
      [
        set carrying_food_amount 0 ;; Eat food immediately and don't carry it to the nest
      ]
      if food = 0 [ set visual_object 0 ]
    ]
  ]
end

;;;
;;; Check if creature returned food to nest and update its fitness
;;;
to compute-creature-fitness
  if last_nest_collision > last_food_collision and last_food_collision > 0 [
    set fitness_value fitness_value + return_food_reward
    set last_nest_collision 0
    set last_food_collision 0
  ]
end

to compute-swarm-fitness
  ask testcreatures [ set fitness_value fitness_value + fitness]
end

;;;
;;; Create photoreceptor and attach it to insect
;;;
to create-visual-sensor [ #psensor_id #pposition #colour_sensitive #attached_neuron_label #attached_creature] ;;Called by observer

  let attached_neuron_id get-input-neuronid-from-label #attached_neuron_label
  create-visualsensors 1 [
     set sensor_id #psensor_id
     set relative_rotation #pposition ;;Degrees relative to current heading - Left + Right 0 Center
     set attached_to_colour #colour_sensitive
     set attached_to_neuron attached_neuron_id
     set attached_to_creature #attached_creature
     ht
  ]

end

;;;
;;; Create pheromonereceptor and attach it to insect
;;;
to create-pheromone-sensor [ #psensor_id #pposition #left_sensor_attached_neuron_label #middle_sensor_attached_neuron_label #right_sensor_attached_neuron_label #attached_creature] ;;Called by observer

  let left_sensor_attached_neuron_id get-input-neuronid-from-label #left_sensor_attached_neuron_label
  let middle_sensor_attached_neuron_id get-input-neuronid-from-label #middle_sensor_attached_neuron_label
  let right_sensor_attached_neuron_id get-input-neuronid-from-label #right_sensor_attached_neuron_label
  create-pheromonesensors 1 [
     set sensor_id #psensor_id
     set relative_rotation #pposition ;;Degrees relative to current heading - Left + Right 0 Center
     set left_sensor_attached_to_neuron left_sensor_attached_neuron_id
     set middle_sensor_attached_to_neuron middle_sensor_attached_neuron_id
     set right_sensor_attached_to_neuron right_sensor_attached_neuron_id
     set attached_to_creature #attached_creature
     ht
  ]

end

;;;
;;; Ask pheromonereceptor if there is pheromone nearby
;;;
to sense-pheromone
  ;;;;;;;;;;;;;;;Take same position and heading of creature:;;;;;;;;;;;;;;;
  let creature_px 0
  let creature_py 0
  let creature_heading 0
  ask  testcreature attached_to_creature [set creature_px xcor set creature_py ycor set creature_heading heading];
  set xcor creature_px
  set ycor creature_py
  set heading creature_heading
  rt relative_rotation
  fd 1
  ;;;;;;;;;;;;;;;Sense
  let scent-ahead chemical-scent-at-angle   0
  let scent-right chemical-scent-at-angle  45
  let scent-left  chemical-scent-at-angle -45
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ifelse (scent-right > scent-ahead) and (scent-right > scent-left)
  [
    ;;activate right neuron
    feed-input-neuron right_sensor_attached_to_neuron 1
  ]
  [
    ifelse (scent-left > scent-ahead) and (scent-left > scent-right)
    [
      ;;activate left neuron
      feed-input-neuron left_sensor_attached_to_neuron 1
    ]
    [
      ;;activate middle neuron
      feed-input-neuron middle_sensor_attached_to_neuron 1
    ]
  ]
end

;;;
;;; Check amount of pheromone at [angle], called by pheromonereceptor
;;;
to-report chemical-scent-at-angle [angle]
  let p patch-right-and-ahead angle 1
  if p = nobody [ report 0 ]
  report [chemical] of p
end

;;;
;;; Create pheromonereceptor and attach it to insect
;;;
to create-nest-sensor [ #psensor_id #pposition #left_sensor_attached_neuron_label #middle_sensor_attached_neuron_label #right_sensor_attached_neuron_label #attached_creature] ;;Called by observer

  let left_sensor_attached_neuron_id get-input-neuronid-from-label #left_sensor_attached_neuron_label
  let middle_sensor_attached_neuron_id get-input-neuronid-from-label #middle_sensor_attached_neuron_label
  let right_sensor_attached_neuron_id get-input-neuronid-from-label #right_sensor_attached_neuron_label
  create-nestsensors 1 [
     set sensor_id #psensor_id
     set relative_rotation #pposition ;;Degrees relative to current heading - Left + Right 0 Center
     set left_sensor_attached_to_neuron left_sensor_attached_neuron_id
     set middle_sensor_attached_to_neuron middle_sensor_attached_neuron_id
     set right_sensor_attached_to_neuron right_sensor_attached_neuron_id
     set attached_to_creature #attached_creature
     ht
  ]

end

;;;
;;; Ask smell-receptor if there is nest scent nearby
;;;
to sense-nest
  ;;;;;;;;;;;;;;;Take same position and heading of creature:;;;;;;;;;;;;;;;
  let creature_px 0
  let creature_py 0
  let creature_heading 0
  ask  testcreature attached_to_creature [set creature_px xcor set creature_py ycor set creature_heading heading];
  set xcor creature_px
  set ycor creature_py
  set heading creature_heading
  rt relative_rotation
  fd 1
  ;;;;;;;;;;;;;;;Sense
  let scent-ahead nest-scent-at-angle   0
  let scent-right nest-scent-at-angle  45
  let scent-left  nest-scent-at-angle -45
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ifelse (scent-right > scent-ahead) and (scent-right > scent-left)
  [
    ;;activate right neuron
    feed-input-neuron right_sensor_attached_to_neuron 1
  ]
  [
    ifelse (scent-left > scent-ahead) and (scent-left > scent-right)
    [
      ;;activate left neuron
      feed-input-neuron left_sensor_attached_to_neuron 1
    ]
    [
      ;;activate middle neuron
      feed-input-neuron middle_sensor_attached_to_neuron 1
    ]
  ]
end

;;;
;;; Check amount of pheromone at [angle], called by pheromonereceptor
;;;
to-report nest-scent-at-angle [angle]
  let p patch-right-and-ahead angle 1
  if p = nobody [ report 0 ]
  report [nest-scent] of p
end

;;;
;;; Ask photoreceptor if there is a patch ahead (within insect_view_distance) with a perceivable colour (= attached_to_colour)
;;;
to view-world-ahead ;;Called by visualsensors

  let itemcount 0
  let foundobj 0
  ;;;;;;;;;;;;;;;Take same position and heading of creature:;;;;;;;;;;;;;;;
  let creature_px 0
  let creature_py 0
  let creature_heading 0
  ask  testcreature attached_to_creature [set creature_px xcor set creature_py ycor set creature_heading heading];
  set xcor creature_px
  set ycor creature_py
  set heading creature_heading
  rt relative_rotation
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  let view_distance insect_view_distance
  let xview 0
  let yview 0
  while [itemcount <= view_distance and foundobj = 0]
  [
    set itemcount itemcount + 0.34;1
    ask patch-ahead itemcount [
       set foundobj visual_object
       set xview pxcor
       set yview pycor
    ]
  ]

  ifelse (foundobj = attached_to_colour) ;;Found perceivable colour?
  [
    set distance_to_stimuli 1 ;Do not take into account the distance. itemcount
    set perceived_stimuli foundobj
  ]
  [
    set distance_to_stimuli 0
    set perceived_stimuli 0
  ]

end

;;;
;;; Process Nociceptive, reward and visual sensation
;;;
to perceive-world ;;Called by testcreatures

 let nextobject 0
 let distobject 0
 let onobject 0
 ;; Get color of current position
 ask patch-here [ set onobject visual_object ] ;;visual_object 1:  type (1= wall, 2= obstacle, 3=food)
 ifelse (onobject = 1)
 [
    ifelse (noxious_white) ;;is White attached to a noxious stimulus
    [
      feed-input-neuron pain_neuron 1 ;;induce Pain
      update-fitness (- collision_punishment)
      if (istrainingmode?)
      [
        set error_free_counter 0
      ]
    ]
    [
        update-fitness sense_food_fitness_reward
        feed-input-neuron reward_neuron 1 ;;induce happiness
        ask patch-here [ set pcolor black ] ;;Eat patch
    ]
 ]
 [
    ifelse (onobject = 2)
    [
      ifelse (noxious_red) ;;is Red attached to a noxious stimulus
      [
        update-fitness (- collision_punishment)
        feed-input-neuron pain_neuron 1 ;;induce Pain
        if (istrainingmode?)
        [
          set error_free_counter 0
        ]
      ]
      [
          update-fitness sense_food_fitness_reward
          feed-input-neuron reward_neuron 1 ;;induce happiness
          ask patch-here [ set pcolor black ]  ;;Eat patch
      ]
    ]
    [
      if (onobject = 3)
      [
         ifelse (noxious_green) ;;is Green attached to a noxious stimulus
         [
           update-fitness (- collision_punishment)
           feed-input-neuron pain_neuron 1 ;;induce Pain
           if (istrainingmode?)
           [
             set error_free_counter 0
           ]
         ]
         [
             update-fitness sense_food_fitness_reward
             feed-input-neuron reward_neuron 1 ;;induce happiness
         ]
      ]
    ]
 ]

end


;;;
;;; Move or rotate according to the active motoneuron
;;;
to do-actuators ;;Called by Creature

 let dorotation_left? false
 let dorotation_right? false
 let domovement? false
 let dopheromone? false
 ;;Check rotate left actuator
  if ([nlast-firing-time] of normalneuron rotate_left_neuron ) = ticks [ set dorotation_left? true ]

 ;;Check rotate right actuator
 if ([nlast-firing-time] of normalneuron rotate_right_neuron ) = ticks [ set dorotation_right? true ]

 ;;Check move forward actuator
 if ([nlast-firing-time] of normalneuron move_neuron ) = ticks [ set domovement? true ]

 ;;Check drop pheromone actuator
 if ([nlast-firing-time] of normalneuron pheromone_neuron ) = ticks [ set dopheromone? true ]

 if (dorotation_left?)
 [
   set fitness_value fitness_value - rotation_actuator_cost ;0.003
    rotate-creature -6
 ]

 if (dorotation_right?)
 [
   set fitness_value fitness_value - rotation_actuator_cost
    rotate-creature 6
 ]

 if (domovement?)
 [
   set fitness_value fitness_value - move_actuator_cost
   move-creature 0.6 ;0.6
 ]

 if (dopheromone?)
 [
   set fitness_value fitness_value - pheromone_cost
   set chemical chemical + 60
 ]

end

;;;
;;; Photoreceptor excitates the connected input neuron
;;;
to propagate-visual-stimuli ;;Called by visual sensor

  if (attached_to_colour = perceived_stimuli) ;;Only produce an action potential if the corresponding associated stimulus was sensed
  [
     feed-input-neuron attached_to_neuron distance_to_stimuli;
  ]

end

;;;
;;; Move insect (#move_units) patches forward
;;;
to move-creature [#move_units]
  fd #move_units
end

;;;
;;; Rotate insect (#rotate_units) degrees
;;;
to rotate-creature [#rotate_units]
  rt #rotate_units
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;End of insect related code;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
to-report parse_csv_line [ #csv-line ]
  let parameter_end true
  let parameters_line #csv-line
  let current_parameter ""
  let parameters_list[]
  while [ parameter_end != false ]
  [
    set parameter_end position "," parameters_line
    if ( parameter_end != false ) [
      set current_parameter substring parameters_line 0 parameter_end
      set parameters_line substring parameters_line ( parameter_end + 1 ) ( (length parameters_line));
      ;;add current_parameter to destination list
      ;parameters_list
      set parameters_list lput (read-from-string current_parameter) parameters_list
    ]
  ]
  if ( length parameters_line ) > 0 [ ;;Get last parameter
    ;;add current_parameter to destination list
    set parameters_list lput (read-from-string parameters_line) parameters_list
  ]

 report parameters_list
end

to setup_neural_parameters

  ;;;;;;;;;;;;;;Setup Neuron types;;;;;;;;;;;;;;
  ;;;;;;;;;;;;;;;;Neuron type1:
  setup-neurontype 1 -65 -55 0.08 -75 3 -75
  set-neurontype-learning-params 1 0.045 75 0.045 -50 15 1 (8 + 3) (15 + 3) ;[ #pneurontypeid #ppos_hebb_weight #ppos_time_window #pneg_hebb_weight #pneg_time_window #pmax_synaptic_weight
                                    ;#pmin_synaptic_weight #ppos_syn_change_interval #pneg_syn_change_interval]


  ;;;;;;;;;;;;;;;;Neuron type2:
  setup-neurontype 2 -65 -55 0.08 -70 1 -70 ;(typeid restpot threshold decayr refractpot refracttime)
  set-neurontype-learning-params 2 0.09 55 0.09 -25 15 1 8 15

end

to build_neural_circuit [ #insect_offset #weight_data #plasticity_data #delay_data ]

  let visual_neuron_red 11
  let visual_input_neuron_red 1
  let visual_neuron_green 12
  let visual_input_neuron_green 2

  let pheromone_neuron_left 13
  let pheromone_input_neuron_left 3
  let pheromone_neuron_middle 14
  let pheromone_input_neuron_middle 4
  let pheromone_neuron_right 15
  let pheromone_input_neuron_right 5

  let nociceptive_neuron 16
  let nociceptive_input_neuron 6
  let rewarding_neuron 17
  let rewarding_input_neuron 7

  let nest_neuron_left 18
  let nest_input_neuron_left 8
  let nest_neuron_middle 19
  let nest_input_neuron_middle 9
  let nest_neuron_right 20
  let nest_input_neuron_right 10

  let on_nest_neuron 21
  let on_nest_input_neuron 11

  let heart_neuron_1 22
  let heart_neuron_2 23
  let heart_input_neuron 12

  let move_actuator_neuron 300
  let rotate_actuator_left_neuron 301
  let rotate_actuator_right_neuron 302
  let pheromone_actuator_neuron 303

  ;;;;;;;;;;;;;;Layer 1: Visual Afferent neurons with receptors ;;;;;;;;;;;;;;;
  setup-normal-neuron 1 15 10  (#insect_offset + visual_neuron_red) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron 5 10  (#insect_offset + visual_input_neuron_red)  (#insect_offset + visual_neuron_red) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron psynapseefficacy pcoding pnumofspikes]

  setup-normal-neuron 1 15 15  (#insect_offset + visual_neuron_green) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron    5 15  (#insect_offset + visual_input_neuron_green)  (#insect_offset + visual_neuron_green) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron ipsynapseefficacy pcoding pnumofspikes]

  ;;;;;;;;;;;;;;Layer 1b: Pheromone Afferent neurons with receptors ;;;;;;;;;;;;;;;
  ;;Left
  setup-normal-neuron 1 15 20  (#insect_offset + pheromone_neuron_left) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron    5 20  (#insect_offset + pheromone_input_neuron_left)  (#insect_offset + pheromone_neuron_left) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron ipsynapseefficacy pcoding pnumofspikes]
  ;;Front
  setup-normal-neuron 1 15 25  (#insect_offset + pheromone_neuron_middle) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron    5 25  (#insect_offset + pheromone_input_neuron_middle)  (#insect_offset + pheromone_neuron_middle) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron ipsynapseefficacy pcoding pnumofspikes]
  ;;Right
  setup-normal-neuron 1 15 30  (#insect_offset + pheromone_neuron_right) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron    5 30  (#insect_offset + pheromone_input_neuron_right)  (#insect_offset + pheromone_neuron_right) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron ipsynapseefficacy pcoding pnumofspikes]
  ;;;;;;;;;;;;;;; Layer 1c: nociceptive and reward neurons with receptors ;;;;;;;;;;;;;;;;;;;;;;

  ;Nociceptive pathway:
  setup-normal-neuron 1 15 35  (#insect_offset + nociceptive_neuron) 1
  setup-input-neuron   5 35  (#insect_offset + nociceptive_input_neuron)  (#insect_offset + nociceptive_neuron) 20 1 pspikefrequency

  ;Reward pathway:
  setup-normal-neuron 1 15 40  (#insect_offset + rewarding_neuron) 1
  setup-input-neuron   5 40  (#insect_offset + rewarding_input_neuron)  (#insect_offset + rewarding_neuron) 20 1 pspikefrequency

  ;;;;;;;;;;;;;;Layer 1d: Smell Afferent neurons with receptors ;;;;;;;;;;;;;;;
  ;;Left
  setup-normal-neuron 1 15 45  (#insect_offset + nest_neuron_left) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron    5 45  (#insect_offset + nest_input_neuron_left)  (#insect_offset + nest_neuron_left) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron ipsynapseefficacy pcoding pnumofspikes]
  ;;Front
  setup-normal-neuron 1 15 50  (#insect_offset + nest_neuron_middle) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron    5 50  (#insect_offset + nest_input_neuron_middle)  (#insect_offset + nest_neuron_middle) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron ipsynapseefficacy pcoding pnumofspikes]
  ;;Right
  setup-normal-neuron 1 15 55  (#insect_offset + nest_neuron_right) 1 ;;[layernum pposx pposy pid pneurontypeid]
  setup-input-neuron    5 55  (#insect_offset + nest_input_neuron_right)  (#insect_offset + nest_neuron_right) 20 1 pspikefrequency ;;[pposx pposy pid ppostsynneuron ipsynapseefficacy pcoding pnumofspikes]
  ;;;;;;;;;;;;;;; Layer 1e: Heart ;;;;;;;;;;;;;;;;;;;;;;

  ;;;;;;Create on nest neuron:

  setup-normal-neuron 1 15 59  (#insect_offset + on_nest_neuron) 1
  setup-input-neuron    5 59  (#insect_offset + on_nest_input_neuron)  (#insect_offset + on_nest_neuron) 20 1 pspikefrequency

  ;;;;;;Create heart layer 1e:
  setup-normal-neuron 2 15 1  (#insect_offset + heart_neuron_1) 2
  setup-normal-neuron 2 15 5  (#insect_offset + heart_neuron_2) 2
  setup-synapse  (#insect_offset + heart_neuron_1)  (#insect_offset + heart_neuron_2) 15 excitatory_synapse 2 false ;(no plasticity needed)
  setup-synapse  (#insect_offset + heart_neuron_2)  (#insect_offset + heart_neuron_1) 15 excitatory_synapse 3 false ;(no plasticity needed)
  setup-input-neuron 5 1  (#insect_offset + heart_input_neuron)  (#insect_offset + heart_neuron_1) 20 1 pspikefrequency ;;Voltage clamp to start pacemaker

  ;;;;;;Create neuronal layer 2 (6):
  let neuron_counter 0
  repeat 10 [
    setup-normal-neuron 3 (10 * cos(neuron_counter * 36) + 40) ( 10 * sin(neuron_counter * 36) + 30)  (#insect_offset + 200 + neuron_counter) 1 ;;[layernum pposx pposy pid pneurontypeid]
    set neuron_counter neuron_counter + 1
  ]

  ;;;;;;Create neuronal motor layer 3 (2):
  set neuron_counter 0
  repeat 4 [
    setup-normal-neuron 4 55 ( 22 + neuron_counter * 5)  (#insect_offset + 300 + neuron_counter) 1 ;;[layernum pposx pposy pid pneurontypeid]
    set neuron_counter neuron_counter + 1
  ]


  ;;;;;;;;;;;;;;;;;;;;;;Create Creature and attach neural circuit
  ;; Start insect hearth
  feed-input-neuron_by_label  (#insect_offset + heart_input_neuron) 1
  let patch_xcor 0
  let patch_ycor 0

  ifelse random_start_position? [
    ask one-of simulation_area with [pcolor = black][ set patch_xcor pxcor set patch_ycor pycor]
  ]
  [
    set patch_xcor nestx
    set patch_ycor nesty
  ]
  let creatureid create-creature patch_xcor patch_ycor (#insect_offset + 1) (#insect_offset + rewarding_input_neuron)  (#insect_offset + nociceptive_input_neuron)  (#insect_offset + move_actuator_neuron)  (#insect_offset + rotate_actuator_left_neuron) (#insect_offset + rotate_actuator_right_neuron) (#insect_offset + pheromone_actuator_neuron) (#insect_offset + on_nest_input_neuron);;[#xpos #ypos #creature_id #reward_input_neuron #pain_input_neuron #move_neuron #rotate_neuron_left #rotate_neuron_right #pheromone_neuron]
  ;;;;;;;;;;Create Pheromone sensor
  create-pheromone-sensor (#insect_offset + 1) 0 (#insect_offset + pheromone_input_neuron_left) (#insect_offset + pheromone_input_neuron_middle) (#insect_offset + pheromone_input_neuron_right) creatureid

  create-nest-sensor (#insect_offset + 1) 0 (#insect_offset + nest_input_neuron_left) (#insect_offset + nest_input_neuron_middle) (#insect_offset + nest_input_neuron_right) creatureid

  ;;;;;;;;;;Create Visual sensors and attach neurons to them;;;;;;;;;
  create-visual-sensor  (#insect_offset + 1) 0  1  (#insect_offset + visual_input_neuron_red) creatureid;[ psensor_id pposition colour_sensitive attached_neuron attached_creature]
  create-visual-sensor  (#insect_offset + 2) 0  3  (#insect_offset + visual_input_neuron_green) creatureid;[ psensor_id pposition colour_sensitive attached_neuron attached_creature]

  ;; Activate training mode
  set istrainingmode? true

  ;;;;;;;;;;;;;;;;;;;;Create synapses from layer 1 to layer 2:
  let type_exc_or_inh_synapse excitatory_synapse
  let current_index 0

  let layer_from 0
  let layer_to 0
  repeat 12 [
    set layer_to 0
    repeat 10 [
      let current_weight item current_index #weight_data
      let current_delay item current_index #delay_data
      let current_plasticity item current_index #plasticity_data
      ifelse current_weight >= 0 [
        set type_exc_or_inh_synapse excitatory_synapse
      ]
      [
        set type_exc_or_inh_synapse inhibitory_synapse
      ]

      let plasticity? false
      if current_plasticity = 1
      [
        set plasticity? true
      ]

      setup-synapse  (#insect_offset + 11 + layer_from)  (#insect_offset + 200 + layer_to) abs(current_weight) type_exc_or_inh_synapse (round current_delay) plasticity?
      set current_index current_index + 1
      set layer_to layer_to + 1
    ]
    set layer_from layer_from + 1
  ]

  ;;;;;;;;;;;;;;;;;;;;Create synapses from layer 2 to layer 2:
  set layer_from 0
  set layer_to 0
  repeat 10 [
    set layer_to 0
    repeat 10 [
      let current_weight item current_index #weight_data
      let current_delay item current_index #delay_data
      let current_plasticity item current_index #plasticity_data
      ifelse current_weight >= 0 [
        set type_exc_or_inh_synapse excitatory_synapse
      ]
      [
        set type_exc_or_inh_synapse inhibitory_synapse
      ]

      let plasticity? false
      if current_plasticity = 1
      [
        set plasticity? true
      ]
      if layer_from != layer_to [
        setup-synapse  (#insect_offset + 200 + layer_from)  (#insect_offset + 200 + layer_to) abs(current_weight) type_exc_or_inh_synapse (round current_delay) plasticity?
        set current_index current_index + 1
      ]
      set layer_to layer_to + 1
    ]
    set layer_from layer_from + 1
  ]
  ;;;;;;;;;;;;;;;;;;;;Create synapses from layer 2 to layer 3:
  set layer_from 0
  set layer_to 0
  repeat 10 [
    set layer_to 0
    repeat 4 [
      let current_weight item current_index #weight_data
      let current_delay item current_index #delay_data
      let current_plasticity item current_index #plasticity_data
      ifelse current_weight >= 0 [
        set type_exc_or_inh_synapse excitatory_synapse
      ]
      [
        set type_exc_or_inh_synapse inhibitory_synapse
      ]

      let plasticity? false
      if current_plasticity = 1
      [
        set plasticity? true
      ]

      setup-synapse  (#insect_offset + 200 + layer_from)  (#insect_offset + 300 + layer_to) abs(current_weight) type_exc_or_inh_synapse (round current_delay) plasticity?
      set current_index current_index + 1
      set layer_to layer_to + 1
    ]
    set layer_from layer_from + 1
  ]

end

to update-fitness [ #fitness_change ]
  set fitness_value fitness_value + #fitness_change
end

;;;
;;; Create neural circuit, insect and world
;;;
to setup

  clear-all
  RESET-TICKS
  let time_seed  (read-from-string( substring date-and-time 3 5 ) + 1) * 100
  print time_seed
  random-seed time_seed

  if used_random_seed != 0 [ random-seed used_random_seed ]
  initialize-global-vars
  ;;;;;;;;;;Draw world with white, green and red patches;;;;;;;;
  draw-world
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  let config_file "individual_config.csv"
  read_l2l_config config_file

end

to read_l2l_config [ #filename ]

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  repeat 250 [
    set plasticity_data lput 0 plasticity_data
  ]
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  file-close-all
  carefully [
    file-open #filename
    set raw_weight_data file-read-line
    set raw_delay_data file-read-line

    set weight_data ( parse_csv_line raw_weight_data )
    set delay_data parse_csv_line raw_delay_data
  ]
  [
    print error-message
  ]

  setup_neural_parameters

  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
  ;load_test_brain_data
  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

  let n_insects 15
  let insect_offset 0
  repeat n_insects [
    build_neural_circuit insect_offset weight_data plasticity_data delay_data
    set insect_offset insect_offset + 1000
  ]

end

to write_l2l_results [ #filename ]
  file-close-all
  file-open #filename
  file-print raw_weight_data
  file-print raw_delay_data
  file-print fitness_value
  file-flush
  file-close-all

end

;;;
;;; Generate insect world with 3 types of patches
;;;
to draw-world
   ask patches [set visual_object 0 set nest? false set food 0]
   ;;;;;;;;;;Create a grid of white patches representing walls in the virtual worls
   ask patches with [ pxcor >= min_x_boundary and pycor = min_y_boundary and pxcor <= max_x_boundary ] [ set pcolor red  ]
   ask patches with [ pxcor >= min_x_boundary and pycor = max_y_boundary ] [ set pcolor red ]
   ask patches with [ pycor >= min_y_boundary and pxcor = min_x_boundary and pxcor <= max_x_boundary] [ set pcolor red ]
   ask patches with [ pycor >= min_y_boundary and pxcor = max_x_boundary ] [ set pcolor red ]
   let ccolumns 0

  set simulation_area patches with [ pxcor >= min_x_boundary and pxcor <= max_x_boundary and pycor >= min_y_boundary and pycor <= max_y_boundary ]
  ask n-of 5 simulation_area with [ ( distancexy 82 30) > 10 and ( distancexy 82 30) < 18 ][ set pcolor green set food 1 ]
  ask simulation_area [set visual_object 0 set nest? false]
  ask simulation_area with [ pcolor = red ][set visual_object 1]       ;; type (1= wall, 3= food)
  ask simulation_area with [ pcolor = green ][set visual_object 3]
  ask simulation_area [ setup-food setup-nest ]

end

;;;
;;; Don't allow the insect to go beyond the world boundaries
;;;
to check-boundaries

   if (istrainingmode?)
   [
      set error_free_counter error_free_counter + 1
      if(error_free_counter > required_error_free_iterations)
      [
       set istrainingmode? false
      ]
      ask testcreatures [
         ifelse (xcor < (min_x_boundary + 2)) [
            set xcor ( min_x_boundary + 2)
         ]
         [
            if (xcor > ( max_x_boundary - 2)) [ set xcor ( max_x_boundary - 2) ]
         ]
         ifelse (ycor < (min_y_boundary + 2)) [
            set ycor ( min_y_boundary + 2)
         ]
         [
            if (ycor >  (max_y_boundary - 2)) [ set ycor ( max_y_boundary - 2)]
         ]

      ]
   ]

end

;;;
;;; Run simulation
;;;
to go

 if (awakecreature?)
 [
   ask visualsensors [ view-world-ahead propagate-visual-stimuli ] ;;Feed visual sensors at first
   ask pheromonesensors [ sense-pheromone ] ;;Sniff nearby pheromone
   ask nestsensors [ sense-nest ]
   ask testcreatures [ perceive-world]; sense touch information
   ;]
   do-network-dynamics
   ask testcreatures [do-actuators check-creature-on-nest-or-food compute-creature-fitness]
 ]
 if ticks mod 7 = 0 [
   diffuse chemical (diffusion-rate / 100)
   ask simulation_area
   [
     set chemical chemical * (100 - evaporation-rate) / 100  ;; slowly evaporate chemical
     recolor-patch
   ]
 ]
 check-boundaries
 update-fitness ( - resting_cost )
 if ticks = 1000  [
    write_l2l_results "individual_result.csv"
    set simulation_end_signal true
    stop
  ]
 tick

end
@#$#@#$#@
GRAPHICS-WINDOW
154
23
970
520
-1
-1
8.0
1
10
1
1
1
0
0
0
1
0
100
0
60
0
0
1
ticks
30.0

BUTTON
9
27
84
61
Setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
10
102
130
136
go one-step
go
NIL
1
T
OBSERVER
NIL
G
NIL
NIL
1

BUTTON
10
65
130
99
go-forever
go
T
1
T
OBSERVER
NIL
S
NIL
NIL
1

SWITCH
5
238
142
271
awakecreature?
awakecreature?
0
1
-1000

SWITCH
6
495
129
528
noxious_red
noxious_red
0
1
-1000

SWITCH
6
529
129
562
noxious_white
noxious_white
0
1
-1000

SWITCH
6
564
129
597
noxious_green
noxious_green
1
1
-1000

SWITCH
5
278
144
311
istrainingmode?
istrainingmode?
0
1
-1000

BUTTON
10
178
130
214
re-draw World
draw-world
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
10
139
130
175
relocate insect
;ask patches with [ pxcor = 102 and pycor = 30 ] [set pcolor green]\nask testcreatures [setxy 82 30]
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
3
316
151
349
insect_view_distance
insect_view_distance
1
15
14.0
1
1
NIL
HORIZONTAL

SWITCH
3
365
152
398
enable_plots?
enable_plots?
1
1
-1000

MONITOR
1372
21
1499
66
NIL
fitness_value
2
1
11

SLIDER
3
408
150
441
evaporation-rate
evaporation-rate
0
20
1.0
1
1
NIL
HORIZONTAL

SLIDER
3
442
150
475
diffusion-rate
diffusion-rate
0.0
99.0
20.0
1.0
1
NIL
HORIZONTAL

SLIDER
988
210
1253
243
sense_food_fitness_reward
sense_food_fitness_reward
0.000
1.0
1.0
0.001
1
NIL
HORIZONTAL

SLIDER
989
253
1252
286
collision_punishment
collision_punishment
0
500
0.0
0.5
1
NIL
HORIZONTAL

SWITCH
992
413
1256
446
bring_food_to_nest?
bring_food_to_nest?
0
1
-1000

INPUTBOX
993
460
1103
520
used_random_seed
0.0
1
0
Number

SLIDER
989
164
1251
197
move_actuator_cost
move_actuator_cost
0
1
0.25
0.05
1
NIL
HORIZONTAL

SLIDER
986
71
1247
104
pheromone_cost
pheromone_cost
0
1
0.05
0.05
1
NIL
HORIZONTAL

SLIDER
987
29
1248
62
resting_cost
resting_cost
0
10
0.5
0.05
1
NIL
HORIZONTAL

SLIDER
990
295
1252
328
food_source_size
food_source_size
1
6
3.0
1
1
NIL
HORIZONTAL

SLIDER
990
335
1252
368
nest_size
nest_size
1
6
2.0
1
1
NIL
HORIZONTAL

SLIDER
987
116
1249
149
rotation_actuator_cost
rotation_actuator_cost
0
1
0.02
0.01
1
NIL
HORIZONTAL

SWITCH
992
374
1253
407
random_start_position?
random_start_position?
1
1
-1000

SLIDER
1283
117
1480
150
return_food_reward
return_food_reward
0
500
220.0
10
1
NIL
HORIZONTAL

@#$#@#$#@
## WHAT IS IT?

The presented Spiking Neural Network (SNN) model is built in the framework of generalized Integrate-and-fire models which recreate to some extend the phenomenological dynamics of neurons while abstracting the biophysical processes behind them. The Spike Timing Dependent Plasticity (STDP) learning approach proposed by (Gerstner and al. 1996, Kempter et al. 1999)  has been implemented and used as the underlying learning mechanism for the experimental neural circuit.

The neural circuit implemented in this model enables a simulated agent representing a virtual insect to move in a two dimensional world, learning to visually identify and avoid noxious stimuli while moving towards perceived rewarding stimuli. At the beginning, the agent is not aware of which stimuli are to be avoided or followed. Learning occurs through  Reward-and-punishment classical conditioning. Here the agent learns to associate different colours with unconditioned reflex responses.

## HOW IT WORKS

The experimental virtual-insect is able to process three types of sensorial information: (1) visual, (2) pain and (3) pleasant or rewarding sensation.  The visual information  is acquired through three photoreceptors where each one of them is sensitive to one specific color (white, red or green). Each photoreceptor is connected with one afferent neuron which propagates the input pulses towards two Motoneurons identified by the labels 21 and 22. Pain is elicited by a nociceptor (labeled 4) whenever the insect collides with a wall or a noxious stimulus. A rewarding or pleasant sensation is elicited by a pheromone (or nutrient smell) sensor (labeled 5) when the insect gets in direct contact with the originating stimulus.

The motor system allows the virtual insect to move forward (neuron labeled 31) or rotate in one direction (neuron labeled 32) according to the reflexive behaviour associated to it. In order to keep the insect moving even in the absence of external stimuli, the motoneuron 22 is connected to a neural oscillator sub-circuit composed of two neurons (identified by the labels 23 and 24)  performing the function of a pacemaker which sends a periodic pulse to Motoneuron 22. The pacemaker is initiated by a pulse from an input neuron (labeled 6) which represents an external input current (i.e; intracellular electrode).

## HOW TO USE IT

1. Press Setup to create:
 a. the neural circuit (on the left of the view)
 b. the insect and its virtual world (on the right of the view)

2. Press go-forever to continually run the simulation.

3. Press re-draw world to change the virtual world by adding random patches.

4. Press relocate insect to bring the insect to its initial (center) position.

5. Use the awake creature switch to enable or disable the movement of the insect.

6. Use the colours switches to indicate which colours are associated to harmful stimuli.

7. Use the insect_view_distance slider to indicate the number of patches the insect can
   look ahead.

8. Use the leave_trail_on? switch to follow the movement of the insect.


On the circuit side:
Input neurons are depicted with green squares.
Normal neurons are represented with pink circles. When a neuron fires its colour changes to red for a short time (for 1 tick or iteration).
Synapses are represented by links (grey lines) between neurons. Inhibitory synapses are depicted by red lines.

If istrainingmode? is on then the training phase is active. During the training phase, the insect is moved one patch forward everytime it is on a patch associated with a noxious stimulus. Otherwise, the insect would keep rotating over the noxious patch. Also, the insect is repositioned in its initial coordinates every time it reaches the virtual-world boundaries.


## THINGS TO NOTICE

At the beginning the insect moves along the virtual-world in a seemingly random way colliding equally with all types of coloured patches. This demonstrates the initial inability of the insect to discriminate and react in response to visual stimuli.  However, after a few thousands iterations (depending on the learning parameters), it can be seen that the trajectories lengthen as the learning of the insect progresses and more obstacles (walls and harmful stimuli) are avoided.

## THINGS TO TRY

Follow the dynamic of single neurons by monitoring the membrane potential plots while running the simulation step by step.

Set different view distances to see if the behaviour of the insect changes.

Manipulate the STDP learning parameters. Which parameters speed up or slow down the adaptation to the environment?

## EXTENDING THE MODEL

Use different kernels for the decay() and epsp() functions to make the model more accurate in biological terms.

## NETLOGO FEATURES

Use of link and list primitives.


## CREDITS AND REFERENCES

If you mention this model in a publication, we ask that you include these citations for the model itself and for the NetLogo software:


* Cristian Jimenez-Romero, David Sousa-Rodrigues, Jeffrey H. Johnson, Vitorino Ramos
 A Model for Foraging Ants, Controlled by Spiking Neural Networks and Double Pheromonesin UK Workshop on Computational Intelligence 2015, University of Exeter, September 2015.


* Wilensky, U. (1999). NetLogo. http://ccl.northwestern.edu/netlogo/. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.2.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
<experiments>
  <experiment name="experiment1" repetitions="1" sequentialRunOrder="false" runMetricsEveryStep="false">
    <setup>setup</setup>
    <go>go</go>
    <exitCondition>simulation_end_signal</exitCondition>
    <metric>fitness_value</metric>
  </experiment>
</experiments>
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
