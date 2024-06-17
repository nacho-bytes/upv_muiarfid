breed [humans human]
breed [statistic_agents statistic_agent]

globals [
  age_group_0_9
  age_group_10_19
  age_group_20_29
  age_group_30_39
  age_group_40_49
  age_group_50_59
  age_group_60_69
  age_group_70_79
  age_group_80
  elapsed-day-hours
  medical_care_used
  number_of_deaths
  death_list
  city_area_patches
  roads_area_patches
  change_lockdown_condition?
  cumulative_infected
  last_cumulative
  cumulative_aware_of_infection
  last_cumulative_aware_of_infection
  logged_transmitters
]

humans-own [
  infected?
  infection-length
  aggravated_symptoms_day
  age-group
  ontreatment?
  gotinfection?
  contagion-chance
  death-chance
  ongoing-infection-hours
  symptoms_delay
  aware_of_infection?
  infectedby
]

statistic_agents-own [
  age-group
  recovered
  deaths
]

patches-own [
  original_map_color
]

to-report calculate_R0
  let list_of_transmitters remove-duplicates logged_transmitters
  let current_case 0
  let sum_repetitions 0
  foreach list_of_transmitters [
    patient -> set current_case patient
    let transmitter_repeated length filter [ i -> i = current_case] logged_transmitters
    set sum_repetitions sum_repetitions + transmitter_repeated
  ]
  ifelse length list_of_transmitters > 0 [
    report sum_repetitions / ( length list_of_transmitters )
  ]
  [
    report 0
  ]

end

;;;;;;;;;;;;;;;;;;;;;;;;;;;   HUMANS PROCEDURES   ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to infection_exposure
  if (not gotinfection?) [
    let people_around humans-on neighbors
    let infected_around people_around with [infected? = true and not ontreatment? and ( ongoing-infection-hours > (average_days_for_contagion * 24)) ]
    let number_of_infected_around count infected_around
    if number_of_infected_around > 0 [
      let within_contagion_distance (random(metres_per_patch) + 1) ;; Assuming each patch represents up to metres_per_patch distance units (meters)
      set within_contagion_distance within_contagion_distance + random-float ( keep_social_distancing ) ;; Assuming the ordered social distance is not followed accurately but more randomly and proportionate to the distance
      ;;;;;;;;; Chance of Contagion according to age group:
      if (contagion-chance >= (random(100) + 1) and within_contagion_distance <= maximum_contagion_distance) [ ;;Assuming the maximum transmission distance is 2 meters
        let transmitter_person nobody
        ask one-of infected_around [ set transmitter_person who ]
        set logged_transmitters lput transmitter_person logged_transmitters
          if length ( logged_transmitters ) > 800 [ ;;Do not allow the list to grow without end, delete older elements.
            set logged_transmitters but-first logged_transmitters
          ]
        get_infected
      ]
    ]

  ]
end

to get_infected
  set color red
  set size 3
  set infected? true
  set gotinfection? true
  set infection-length 24 * ( random-normal average_infection_length 5.0 ) ;; mean of infection length and standard-deviation multiplied by 24 hours
  set aggravated_symptoms_day round (infection-length / 2.5) ;; Aggravated infection may happen after the first week
  set symptoms_delay 24 * ( random-normal average-symptoms-show 4.0 )
  set ongoing-infection-hours 0
  set cumulative_infected cumulative_infected + 1
end

to get-healthy
  set infected? false
  set gotinfection? true
  set infection-length 0
  set ongoing-infection-hours 0
  set aggravated_symptoms_day 0
  if ontreatment? [ free-medical-care set ontreatment? false]
  set color green
  set size 1
  set aware_of_infection? false
  update-recovered-statistics age-group
end

to check_health_state
  if infected? [
    if ongoing-infection-hours >= symptoms_delay and not ontreatment? [
      if not aware_of_infection? [
        set aware_of_infection? true
        set cumulative_aware_of_infection cumulative_aware_of_infection + 1
      ]
      ifelse prioritise_elderly? [
        ifelse age-group >= age_group_60_69 [
          if get-medical-care = true [
            set ontreatment? true
          ]
        ]
        [
          if %medical-care-availability >= 25 [ ;;If not an elderly person then only take medical care if availability >= 25%
            if get-medical-care = true [
              set ontreatment? true
            ]
          ]
        ]
      ]
      [
        if get-medical-care = true [
          set ontreatment? true
        ]
      ]
    ]
    if (ongoing-infection-hours = aggravated_symptoms_day) ;;Check if patient is going to die
    [
      ;;;;;;;;;; Decide if patient survived or not the infection:
      let chance_to_die 0
      let severity_factor 1
      if ( ( chance_of_severe_infection * 1000 ) >= random(100000) ) [ ;;Patient got a severe infection increasing the death chance by a severity factor
        set severity_factor severity_death_chance_multiplier
      ]
      ifelse (ontreatment?) [
        set chance_to_die ((death-chance * 1000) * severity_factor) * 0.5 ;; Death chance is reduced to 50%, Survival chance is increased by 50%
      ]
      [
        set chance_to_die (death-chance * 1000) * severity_factor
      ]

      if (chance_to_die >= (random(100000) + 1)) [
        update-death-statistics age-group
        set number_of_deaths number_of_deaths + 1
        delete-person
      ]
    ]

    ifelse (ongoing-infection-hours >= infection-length)
    [
      set ongoing-infection-hours 0
      get-healthy
    ]
    [
      set ongoing-infection-hours ongoing-infection-hours + 1
    ]
  ]
end

to move [ #speed ]
  if not ontreatment?
  [
    rt random-float 360
    let next_patch_color white
    ask patch-ahead 1 [ set next_patch_color original_map_color ]
    if (next_patch_color = white ) [ fd #speed ]
  ]
end

to delete-person
  if ontreatment? [ free-medical-care ]
  die
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;   SETUP PROCEDURES   ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to draw_road_lines
  let rows 1
  repeat 8 [
    ask patches with [pxcor > -260 and pxcor < 260 and pycor = -250 + (rows * 60) and pcolor = white ][set pcolor yellow]
    let roads 1
    repeat 3 [
      ask patches with [pxcor = -260 + (rows * 60) + roads and pycor > -250 and pycor < 260 and pcolor = white ][set pcolor yellow]
      set roads roads + 1
    ]
    set rows rows + 1
  ]
end

to create_city_map
  ask patches with [pxcor > -270 and pxcor < 270 and pycor > -270 and pycor < 260 ] [ set pcolor white ]
  let rows 1
  repeat 8 [
    ask patches with [pxcor > -260 and pxcor < 260 and pycor = -250 + (rows * 60) ][set pcolor yellow]
    let roads 1
    repeat 3 [
      ask patches with [pxcor = -260 + (rows * 60) + roads and pycor > -250 and pycor < 260 ][set pcolor yellow]
      set roads roads + 1
    ]
    set rows rows + 1
  ]
end

to setup-globals
  ifelse load_city_map? [
    import-pcolors "Argentina_outline.png"
    draw_road_lines
  ]
  [
    create_city_map
  ]
  ask patches [ set original_map_color pcolor ]
  set age_group_0_9 9
  set age_group_10_19 19
  set age_group_20_29 29
  set age_group_30_39 39
  set age_group_40_49 49
  set age_group_50_59 59
  set age_group_60_69 69
  set age_group_70_79 79
  set age_group_80  80
  set elapsed-day-hours 0
  set medical_care_used 0
  set number_of_deaths 0
  set cumulative_infected 0
  set last_cumulative 0
  set city_area_patches patches with [ pcolor != black ]
  set roads_area_patches patches with [ pcolor = yellow ]
  set complete_lockdown? false
  set change_lockdown_condition? false
  set prioritise_elderly? false
  set partial_lockdown? false
  set cumulative_aware_of_infection 0
  set last_cumulative_aware_of_infection 0
  set logged_transmitters[]
end

to setup_statistic_agent [ #age-group ]
  create-statistic_agents 1
  [
    set age-group #age-group
    set recovered 0
    set deaths 0
    ht
  ]
end

to setup-people [#number #age-group]
  create-humans #number
    [
      let random_x 0
      let random_y 0
      ask one-of city_area_patches [ set random_x pxcor set random_y pycor ]
      setxy random_x random_y
      set shape "circle"
      set infected? false
      set aggravated_symptoms_day 0
      set ongoing-infection-hours 0
      set color green
      set age-group #age-group
      set ontreatment? false
      set gotinfection? false
      set symptoms_delay 0
      set aware_of_infection? false
      set infectedby nobody

      ifelse age-group <= age_group_0_9 [
        set contagion-chance chance_of_infection_0-9
        set death-chance chance_of_death_0-9
      ]
      [
        ifelse age-group <= age_group_10_19 [
          set contagion-chance chance_of_infection_10-19
          set death-chance chance_of_death_10-19
        ]
        [
          ifelse age-group <= age_group_20_29 [
            set contagion-chance chance_of_infection_20-29
            set death-chance chance_of_death_20-29
          ]
          [
            ifelse age-group <= age_group_30_39 [
              set contagion-chance chance_of_infection_30-39
              set death-chance chance_of_death_30-39
            ]
            [
              ifelse age-group <= age_group_40_49 [
                set contagion-chance chance_of_infection_40-49
                set death-chance chance_of_death_40-49
              ]
              [
                ifelse age-group <= age_group_50_59 [
                  set contagion-chance chance_of_infection_50-59
                  set death-chance chance_of_death_50-59
                ]
                [
                  ifelse age-group <= age_group_60_69 [
                    set contagion-chance chance_of_infection_60-69
                    set death-chance chance_of_death_60-69
                  ]
                  [
                    ifelse age-group <= age_group_70_79 [
                      set contagion-chance chance_of_infection_70-79
                      set death-chance chance_of_death_70-79
                    ]
                    [
                        set contagion-chance chance_of_infection_80
                        set death-chance chance_of_death_80
                    ]
                  ]
                ]
              ]
            ]
          ]

        ]
      ]
     ]
end

to setup
  clear-all
  ;;random-seed 45685
  setup-globals
  setup-people population_0-9 age_group_0_9
  setup-people population_10-19 age_group_10_19
  setup-people population_20-29 age_group_20_29
  setup-people population_30-39 age_group_30_39
  setup-people population_40-49 age_group_40_49
  setup-people population_50-59 age_group_50_59
  setup-people population_60-69 age_group_60_69
  setup-people population_70-79 age_group_70_79
  setup-people population_80 age_group_80
  let affected_number round (count humans * (initial_infected_population / 100))
  infect_people affected_number

  setup_statistic_agent age_group_0_9
  setup_statistic_agent age_group_10_19
  setup_statistic_agent age_group_20_29
  setup_statistic_agent age_group_30_39
  setup_statistic_agent age_group_40_49
  setup_statistic_agent age_group_50_59
  setup_statistic_agent age_group_60_69
  setup_statistic_agent age_group_70_79
  setup_statistic_agent age_group_80
  reset-ticks
end

;;;;;;;;;;;;;;;;;;;;;;;;;   ENVIRONMENT - STATISTIC_AGENTS PROCEDURES   ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to update-recovered-statistics [ #age-group ]

  ask statistic_agents with [ age-group = #age-group ] [ set recovered recovered + 1 ]

end

to update-death-statistics [ #age-group ]

  ask statistic_agents with [ age-group = #age-group ] [ set deaths deaths + 1 ]

end

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;   ENVIRONMENT - HUMAN PROCEDURES   ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to infect_people [#affected_number]
  ask n-of #affected_number humans with [ not gotinfection? ] [ get_infected ]
end

to-report get-medical-care
  if medical_care_used < medical_care_capacity [
    set medical_care_used medical_care_used + 1
    report true
  ]
  report false
end

to-report %medical-care-availability
  report ( (medical_care_capacity - medical_care_used) / medical_care_capacity ) * 100
end


to free-medical-care
  set medical_care_used medical_care_used - 1
end

to people-enter-city [#people_entering #percentage_entering_infected_population]
  let entering_per_age_group #people_entering / 9
  setup-people entering_per_age_group age_group_0_9
  setup-people entering_per_age_group age_group_10_19
  setup-people entering_per_age_group age_group_20_29
  setup-people entering_per_age_group age_group_30_39
  setup-people entering_per_age_group age_group_40_49
  setup-people entering_per_age_group age_group_50_59
  setup-people entering_per_age_group age_group_60_69
  setup-people entering_per_age_group age_group_70_79
  setup-people entering_per_age_group age_group_80
  infect_people #people_entering * #percentage_entering_infected_population / 100
end

to people-leave-city [#people_leaving]
  ask n-of #people_leaving humans with [not ontreatment?] [ delete-person ]
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;   SOCIAL INTERACTIONS - HUMANS PROCEDURES   ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to gather_in_schools
  if partial_lockdown? [ stop ]
  let gatherings_size 200 ;Students per school 200
  repeat active_schools [
    let place_x 0
    let place_y 0
    ask one-of city_area_patches [ set place_x pxcor set place_y pycor]
    ask up-to-n-of gatherings_size humans with [ ( age-group = age_group_0_9 or age-group = age_group_10_19 ) and not aware_of_infection? ][set xcor place_x - 4 + random(8) set ycor place_y - 4 + random(8) ]
  ]
end

to gather_in_colleges
  if partial_lockdown? [ stop ]
  let gatherings_size 300 ;Students per college
  repeat active_colleges [
    let place_x 0
    let place_y 0
    ask one-of city_area_patches [ set place_x pxcor set place_y pycor]
    ask up-to-n-of gatherings_size humans with [ age-group = age_group_20_29 and not aware_of_infection? ][set xcor place_x - 4 + random(8) set ycor place_y - 4 + random(8) ]
  ]
end

to gather_in_hosp_venues
  if partial_lockdown? [ stop ]
  let gatherings_size 40 ;40 people per venue
  repeat active_hosp_venues [
    let place_x 0
    let place_y 0
    ask one-of city_area_patches [ set place_x pxcor set place_y pycor]
    ask up-to-n-of gatherings_size humans with [ not aware_of_infection? ][set xcor place_x - 2 + random(3) set ycor place_y - 2 + random(3) ]
  ]
end

to gather_in_public_transport_lines
  if partial_lockdown? [ stop ]
  let gatherings_size 60 ;60 people per bus/tram line
  repeat active_public_transport_lines [
    let place_x 0
    let place_y 0
    ask one-of roads_area_patches [ set place_x pxcor set place_y pycor]
    ask up-to-n-of gatherings_size humans with [ not aware_of_infection? ][set xcor place_x - 2 + random(3) set ycor place_y - 4 + random(10) ]
  ]
end

to gather_in_adult_venues
  if partial_lockdown? [ stop ]
  let gatherings_size 40 ; 40 people per venue
  repeat active_adult_venues [
    let place_x 0
    let place_y 0
    ask one-of city_area_patches [ set place_x pxcor set place_y pycor]
    ask up-to-n-of gatherings_size humans with [ ( age-group != age_group_0_9 and age-group != age_group_10_19 ) and not aware_of_infection? ][set xcor place_x - 2 + random(3) set ycor place_y - 2 + random(3) ]
  ]
end

to gather_in_senior_venues
  if partial_lockdown? [ stop ]
  let gatherings_size 40 ; 40 people per venue
  repeat active_adult_venues [
    let place_x 0
    let place_y 0
    ask one-of city_area_patches [ set place_x pxcor set place_y pycor]
    ask up-to-n-of gatherings_size humans with [ ( age-group = age_group_60_69 or age-group = age_group_70_79 or age-group = age_group_80 ) and not aware_of_infection? ][set xcor place_x - 2 + random(3) set ycor place_y - 2 + random(3) ]
  ]
end

to gather_in_food_shops
  if partial_lockdown? [ stop ]
  let gatherings_size 50 ; 50 people in a supermarket, bakery, small shop
  repeat active_adult_venues [
    let place_x 0
    let place_y 0
    ask one-of city_area_patches [ set place_x pxcor set place_y pycor]
    ask up-to-n-of gatherings_size humans with [ age-group != age_group_0_9 and not aware_of_infection? ][set xcor place_x - 2 + random(4) set ycor place_y - 2 + random(4) ]
  ]
end

to go_back_home
  ask humans with [ not aware_of_infection? ]
  [
    let random_x 0
    let random_y 0
    ask one-of city_area_patches [ set random_x pxcor set random_y pycor ]
    setxy random_x random_y
  ]
end

;;;;;;;;;;;;;;;;;;;;;;;;;;;   GO PROCEDURE   ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

to go

  ifelse prioritise_elderly? [
    foreach sort-on [(- age-group)] humans
    [ patient -> ask patient [ check_health_state ] ]
  ]
  [
      ask humans [ check_health_state ]
  ]
  ask humans [
    ;check_health_state
    infection_exposure
    ifelse not complete_lockdown? [ move 1.5 ][ move 0.1 ]
  ]
  ifelse elapsed-day-hours >= 24
  [
    print cumulative_aware_of_infection
    if log_infection_data? [
      ;let delta_cumulative cumulative_infected / (last_cumulative + 1)
      ;print ( word ceiling (ticks / 24) "," cumulative_infected "," number_of_deaths "," delta_cumulative )
      let delta_cumulative cumulative_aware_of_infection / (last_cumulative_aware_of_infection + 1)
      print ( word ceiling (ticks / 24) "," cumulative_infected "," number_of_deaths "," delta_cumulative )
      set last_cumulative_aware_of_infection cumulative_aware_of_infection
      set last_cumulative cumulative_infected
    ]
    if not complete_lockdown? [
      ;;;;; People enter and leave the city once a day:
      people-leave-city people_entering_city_per_day; people_leaving_city_per_day
      people-enter-city people_entering_city_per_day infected_visitors;[#people_entering #percentage_entering_infected_population]
      ;;;;; Gatherings once a day
      gather_in_schools
      gather_in_colleges
      gather_in_adult_venues
      gather_in_senior_venues
    ]
    set elapsed-day-hours 1
  ]
  [
    if not complete_lockdown? [
      if elapsed-day-hours mod 7 = 0 [ gather_in_hosp_venues ] ;;3 times a day
      if elapsed-day-hours mod 10 = 0 [ gather_in_food_shops ] ;;2 times a day
      if elapsed-day-hours mod 2 = 0 [ gather_in_public_transport_lines ] ;; 12 times a day
    ]
    set elapsed-day-hours elapsed-day-hours + 1
  ]
  if change_lockdown_condition? != complete_lockdown? [
    set change_lockdown_condition? complete_lockdown?
    go_back_home
  ]

  tick
end

to-report %infected
  ifelse any? humans
    [ report (count humans with [infected?] / count humans) * 100 ]
    [ report 0 ]
end
@#$#@#$#@
GRAPHICS-WINDOW
599
147
1208
1057
-1
-1
1.0
1
10
1
1
1
0
1
1
1
-300
300
-450
450
1
1
1
Hours
30.0

BUTTON
1219
491
1366
524
setup
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
1384
492
1509
525
go
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
0

MONITOR
691
10
822
59
% People infected
%infected
2
1
12

PLOT
32
409
571
609
Population health
Days
people
0.0
90.0
0.0
350.0
true
true
"set-plot-y-range 0 ((count humans) + 50)" "if not show_plot_1? [stop]"
PENS
"Covid19-" 0.04 0 -10899396 true "" "plot count humans with [not infected?]"
"Covid19+" 0.04 0 -2674135 true "" "plot count humans with [infected?]"
"Recovered" 0.04 0 -8431303 true "" "plot count humans with [not infected? and gotinfection?]"
"Reported" 0.04 0 -13791810 true "" "plot cumulative_aware_of_infection"

SLIDER
210
59
378
92
chance_of_infection_0-9
chance_of_infection_0-9
0
100
10.0
1
1
NIL
HORIZONTAL

SLIDER
210
97
380
130
chance_of_infection_10-19
chance_of_infection_10-19
0
100
10.0
1
1
NIL
HORIZONTAL

SLIDER
210
136
381
169
chance_of_infection_20-29
chance_of_infection_20-29
0
100
10.0
1
1
NIL
HORIZONTAL

SLIDER
210
172
382
205
chance_of_infection_30-39
chance_of_infection_30-39
0
100
10.0
1
1
NIL
HORIZONTAL

SLIDER
210
210
382
243
chance_of_infection_40-49
chance_of_infection_40-49
0
100
10.0
1
1
NIL
HORIZONTAL

SLIDER
210
248
382
281
chance_of_infection_50-59
chance_of_infection_50-59
0
100
10.0
1
1
NIL
HORIZONTAL

SLIDER
210
287
381
320
chance_of_infection_60-69
chance_of_infection_60-69
0
100
10.0
1
1
NIL
HORIZONTAL

SLIDER
210
326
381
359
chance_of_infection_70-79
chance_of_infection_70-79
0
100
10.0
1
1
NIL
HORIZONTAL

SLIDER
211
364
381
397
chance_of_infection_80
chance_of_infection_80
0
100
10.0
1
1
NIL
HORIZONTAL

SLIDER
407
59
573
92
chance_of_death_0-9
chance_of_death_0-9
0
20
0.1
0.1
1
NIL
HORIZONTAL

SLIDER
407
97
572
130
chance_of_death_10-19
chance_of_death_10-19
0
20
0.2
0.1
1
NIL
HORIZONTAL

SLIDER
407
136
573
169
chance_of_death_20-29
chance_of_death_20-29
0
20
0.2
0.1
1
NIL
HORIZONTAL

SLIDER
407
172
573
205
chance_of_death_30-39
chance_of_death_30-39
0
20
0.2
0.1
1
NIL
HORIZONTAL

SLIDER
407
210
573
243
chance_of_death_40-49
chance_of_death_40-49
0
20
0.4
0.1
1
NIL
HORIZONTAL

SLIDER
407
248
573
281
chance_of_death_50-59
chance_of_death_50-59
0
20
1.3
0.1
1
NIL
HORIZONTAL

SLIDER
408
287
573
320
chance_of_death_60-69
chance_of_death_60-69
0
50
4.6
0.1
1
NIL
HORIZONTAL

SLIDER
407
324
573
357
chance_of_death_70-79
chance_of_death_70-79
0
50
9.8
0.1
1
NIL
HORIZONTAL

SLIDER
407
364
573
397
chance_of_death_80
chance_of_death_80
0
50
18.0
0.1
1
NIL
HORIZONTAL

SLIDER
24
62
177
95
population_0-9
population_0-9
0
10000
3300.0
100
1
NIL
HORIZONTAL

SLIDER
24
100
177
133
population_10-19
population_10-19
0
10000
3100.0
100
1
NIL
HORIZONTAL

SLIDER
24
138
178
171
population_20-29
population_20-29
0
10000
3100.0
100
1
NIL
HORIZONTAL

SLIDER
24
174
179
207
population_30-39
population_30-39
0
10000
2900.0
100
1
NIL
HORIZONTAL

SLIDER
24
213
178
246
population_40-49
population_40-49
0
10000
2500.0
100
1
NIL
HORIZONTAL

SLIDER
24
251
179
284
population_50-59
population_50-59
0
10000
1900.0
100
1
NIL
HORIZONTAL

SLIDER
24
290
178
323
population_60-69
population_60-69
0
10000
1600.0
100
1
NIL
HORIZONTAL

SLIDER
24
328
178
361
population_70-79
population_70-79
0
10000
1000.0
100
1
NIL
HORIZONTAL

SLIDER
24
366
178
399
population_80
population_80
0
10000
1600.0
100
1
NIL
HORIZONTAL

TEXTBOX
7
34
266
66
Initial population by age group
12
0.0
1

TEXTBOX
187
36
401
66
%Chance of infection by age group
12
0.0
1

TEXTBOX
401
36
589
66
%Chance of death by age group
12
0.0
1

SLIDER
1227
79
1427
112
initial_infected_population
initial_infected_population
0
1
0.04
0.01
1
%
HORIZONTAL

SLIDER
1227
118
1427
151
average_infection_length
average_infection_length
1
60
26.0
1
1
Days
HORIZONTAL

SLIDER
1447
298
1651
331
people_entering_city_per_day
people_entering_city_per_day
0
1000
0.0
1
1
NIL
HORIZONTAL

SLIDER
1447
336
1652
369
infected_visitors
infected_visitors
0
100
0.0
1
1
%
HORIZONTAL

SLIDER
1228
39
1428
72
medical_care_capacity
medical_care_capacity
0
250
100.0
1
1
Beds
HORIZONTAL

MONITOR
827
63
947
112
medical_care_used
medical_care_used
17
1
12

SLIDER
1228
158
1427
191
average-symptoms-show
average-symptoms-show
4
15
10.0
1
1
Days
HORIZONTAL

TEXTBOX
1238
353
1442
437
Assumptions:\n- Survival chance when on treatment is increased by 50%\n- Patients on treatment are isolated and not contagious.\n- Infection can not be contracted twice.\n- Each patch in the map is equivalent to a distance of upto 5 meters.\n
11
0.0
1

SLIDER
1445
39
1650
72
active_schools
active_schools
0
30
6.0
1
1
NIL
HORIZONTAL

SLIDER
1445
77
1649
110
active_colleges
active_colleges
0
30
2.0
1
1
NIL
HORIZONTAL

MONITOR
949
63
1068
112
number_of_deaths
number_of_deaths
2
1
12

SLIDER
1445
115
1650
148
active_hosp_venues
active_hosp_venues
0
50
12.0
1
1
NIL
HORIZONTAL

SLIDER
1446
150
1650
183
active_adult_venues
active_adult_venues
0
30
3.0
1
1
NIL
HORIZONTAL

SLIDER
1446
226
1650
259
active_public_transport_lines
active_public_transport_lines
0
100
10.0
1
1
NIL
HORIZONTAL

SLIDER
1446
263
1649
296
active_food_shops
active_food_shops
0
30
6.0
1
1
NIL
HORIZONTAL

SLIDER
1228
197
1428
230
average_days_for_contagion
average_days_for_contagion
0
100
4.0
1
1
days
HORIZONTAL

SWITCH
1217
542
1369
575
complete_lockdown?
complete_lockdown?
1
1
-1000

PLOT
34
634
573
834
Population infection per age group
Days
people
0.0
90.0
0.0
20.0
true
true
";set-plot-y-range 0 ((count humans / 9) + 50)" "if not show_plot_2? [stop]"
PENS
"0-9" 0.04 0 -16777216 true "" "plot count humans with [age-group = age_group_0_9 and infected?]"
"10-19" 0.04 0 -7500403 true "" "plot count humans with [age-group = age_group_10_19 and infected?]"
"20-29" 0.04 0 -2674135 true "" "plot count humans with [age-group = age_group_20_29 and infected?]"
"30-39" 0.04 0 -955883 true "" "plot count humans with [age-group = age_group_30_39 and infected?]"
"40-49" 0.04 0 -6459832 true "" "plot count humans with [age-group = age_group_40_49 and infected?]"
"50-59" 0.04 0 -1184463 true "" "plot count humans with [age-group = age_group_50_59 and infected?]"
"60-69" 0.04 0 -10899396 true "" "plot count humans with [age-group = age_group_60_69 and infected?]"
"70-79" 0.04 0 -13840069 true "" "plot count humans with [age-group = age_group_70_79 and infected?]"
">=80" 0.04 0 -14835848 true "" "plot count humans with [age-group = age_group_80 and infected?]"

PLOT
32
867
574
1067
Deaths per age group
Days
people
0.0
90.0
0.0
20.0
true
true
"" "if not show_plot_3? [stop]"
PENS
"0-9" 0.04 0 -16777216 true "" "plot [deaths] of one-of statistic_agents with [age-group = age_group_0_9]"
"10-19" 0.04 0 -7500403 true "" "plot [deaths] of one-of statistic_agents with [age-group = age_group_10_19]"
"20-29" 0.04 0 -2674135 true "" "plot [deaths] of one-of statistic_agents with [age-group = age_group_20_29]"
"30-39" 0.04 0 -955883 true "" "plot [deaths] of one-of statistic_agents with [age-group = age_group_30_39]"
"40-49" 0.04 0 -6459832 true "" "plot [deaths] of one-of statistic_agents with [age-group = age_group_40_49]"
"50-59" 0.04 0 -1184463 true "" "plot [deaths] of one-of statistic_agents with [age-group = age_group_50_59]"
"60-69" 0.04 0 -10899396 true "" "plot [deaths] of one-of statistic_agents with [age-group = age_group_60_69]"
"70-79" 0.04 0 -13840069 true "" "plot [deaths] of one-of statistic_agents with [age-group = age_group_70_79]"
">=80" 0.04 0 -14835848 true "" "plot [deaths] of one-of statistic_agents with [age-group = age_group_80]"

SWITCH
33
833
156
866
show_plot_2?
show_plot_2?
1
1
-1000

SWITCH
33
609
156
642
show_plot_1?
show_plot_1?
0
1
-1000

TEXTBOX
1256
11
1406
29
Medical parameters
12
0.0
1

TEXTBOX
1468
10
1631
40
Social interactions parameters
12
0.0
1

TEXTBOX
1296
460
1441
478
Actions and Orders
16
0.0
1

TEXTBOX
1384
539
1651
581
Complete lockdown stops all social activities including use of public transport and entering/leaving town. People are not allowed to leave their residences.
11
0.0
1

MONITOR
822
10
957
59
# People infected
count humans with [infected?]
0
1
12

TEXTBOX
217
10
367
28
Population parameters
14
0.0
1

MONITOR
613
10
670
71
Days
ceiling (ticks / 24)
0
1
15

MONITOR
691
63
824
112
# People Recovered
count humans with [not infected? and gotinfection?]
0
1
12

SWITCH
30
1059
155
1092
show_plot_3?
show_plot_3?
1
1
-1000

SLIDER
1221
658
1427
691
keep_social_distancing
keep_social_distancing
0
4
0.0
0.5
1
Meters
HORIZONTAL

TEXTBOX
1438
653
1655
723
Assuming the ordered social distance is not followed accurately but more randomly and proportionate to the distance
11
0.0
1

SLIDER
1228
235
1430
268
maximum_contagion_distance
maximum_contagion_distance
1
3
2.0
0.5
1
mts
HORIZONTAL

SLIDER
1229
273
1430
306
chance_of_severe_infection
chance_of_severe_infection
0
100
16.5
0.5
1
%
HORIZONTAL

SLIDER
1229
312
1431
345
severity_death_chance_multiplier
severity_death_chance_multiplier
1
3
1.0
0.1
1
NIL
HORIZONTAL

MONITOR
958
10
1068
59
# untreated
count humans with [infected?] - count humans with [ontreatment?]
0
1
12

SLIDER
1447
188
1652
221
active_senior_venues
active_senior_venues
0
50
1.0
1
1
NIL
HORIZONTAL

SWITCH
1220
712
1372
745
prioritise_elderly?
prioritise_elderly?
1
1
-1000

TEXTBOX
1392
713
1542
741
Elderly have prioritised access to medical care
11
0.0
1

SWITCH
1521
492
1662
525
load_city_map?
load_city_map?
1
1
-1000

SLIDER
1448
379
1653
412
metres_per_patch
metres_per_patch
1
40
20.0
1
1
NIL
HORIZONTAL

SWITCH
1216
594
1370
627
partial_lockdown?
partial_lockdown?
1
1
-1000

MONITOR
1071
10
1205
59
# Cumulative infected
cumulative_infected
0
1
12

SWITCH
1220
760
1372
793
log_infection_data?
log_infection_data?
1
1
-1000

MONITOR
1070
63
1207
112
# aware of infection
cumulative_aware_of_infection
0
1
12

TEXTBOX
1383
592
1662
644
stops all social activities including public transport, commuters entering and leaving the city are still allowed. No restriction of going out of home.
11
0.0
1

MONITOR
613
75
670
140
   R0
calculate_R0
2
1
16

BUTTON
1850
269
1980
303
Run Experiment 1
set complete_lockdown? false\nset partial_lockdown? false\nset keep_social_distancing 0.0\nwhile [ cumulative_aware_of_infection < 128 ] [\n   go\n]\n
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
1853
336
1985
370
Experiment Part 2
print \"------------Start Lockdown--------\"\nprint \"---Day: \"\nprint ceiling (ticks / 24)\nprint \"--------------------\"\n\nset partial_lockdown? true\nset keep_social_distancing 1.0\nwhile [ (ticks / 24) <= 66 ] [\n   go\n]
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

@#$#@#$#@
## WHAT IS IT?

The model presented here is a multi-agent simulation that visualises emerging dynamics from the interaction and influence of a small subset of multiple biological and social factors in the development of the covid-19 pandemic.
Given the scale of the pandemic, and the complexity of global societal structures, the simulation may not be viewed as a precise model of universal application. The simulation may however be tailored to locales to enable governments to assess strategic interventions and outcomes at local, municipal and regional levels. In facilitating such analysis, variables that play a critical role in the development of the pandemic have been singled out for manipulation in the model.
The variables so enabled, broken down by base line variable, are as follows:

-  Population
   Initial population per age group
   Chance of infection per age group
   Chance of death per age group
   Number of people commuting per day
   Percentage of infection in commuters
   Number of square metres represented per patch.

-  Clinical parameters
   Medical care capacity
   Initial infected population
   Average infection length
   Average time until symptoms show
   Average days for contagion to be possible
   Maximum distance for contagion to be possible
   Chance of having a severe infection
   Multipliying factor for chance of death in case of a severe infection

-  Social interaction (being the number of places where groups of people of various age groups meet frequently)
   Number of schools
   Number of colleges
   Number of hospitality venues
   Number of adult venues
   Number of senior venues
   Number of public transport lines
   Number of food shops


## HOW IT WORKS

Every tick of the simulation represents one hour. 
The virtual world area shows a map outline in white (we chose the Republic of San Marino) with main roads outlined in yellow.
Non infected people are represented with green dots on the map.
Infected people are represented with red human shapes.
As the simulation runs, clusters of green dots will appear and dissipate. These correspond to the social gatherings of simulated people from different age groups in different types of venues and activities during the day. People using public transport gather mainly along the yellow lines in the map.
At the top of the map there are monitors indicating the following variables:
- Days: Number of elapsed days since the simulation started.
- % People infected: Percentage of people with ongoing infection.
- # People infected: Number of people with ongoing infection.
- # Untreated: Number of people with infection but not receiving medical care.
- # People recovered:  Number of people recovered from infection since the simulation started.
Medical care used:  Number of hospital beds used.
Number of deaths: Number of people deceased since the simulation started.

Plots:
There are 3 plots at the bottom-left side of the screen. This plots show:
Population health: This plot shows the curves of:
-	Healthy population
-	Population infected with Covid19
-	Population recovered from infection with Covid 19
Population infection with Covid19 per age group:
-	0-9 years
-	10-19 years
-	20-29 years
-	30-39 years
-	40-49 years
-	50-59 years
-	60-69 years
-	70-79 years
-	>= 80 years
Deaths caused by infection with Covid19 per age group:
-	0-9 years
-	10-19 years
-	20-29 years
-	30-39 years
-	40-49 years
-	50-59 years
-	60-69 years
-	70-79 years
-	>= 80 years


## HOW TO USE IT

1.	Scroll right or down as required to access the rest of the user interface.
2.	Press the setup button.
3.	Adjust the sliders of parameters according to the desired values.
4.	Press the go button to start the simulation.
5.	Observe plots at the bottom of the user interface. Plots can be disabled and enabled at anytime. Disabling plots can improve the speed of the simulation.
6.	If there is need to change parameters or execute actions during the simulation, it is recommended to pause the execution using the go button. Change the parameters or execute orders as needed. Press go to continue with execution.



## THINGS TO TRY

Strategizing intervention
In enabling students, scientists and governments to strategize responses and outcomes, the simulation incorporates a so ‘lock down’ switch. That is, to observe the development of the infection when the population is restricted in terms of mobility and interaction in public places.
The simulation has the possibility of being built out to incorporate additional variables, and strategic responses.

Actions and Orders:

1. Setup: reset all variables and set the initial conditions for the simulation.

2. Go: runs the simulation. It can be paused/stop at any time by pressing the “go” button.

3. Complete lockdown: Per default is Off. When set to On it tells the simulation to send everyone home, stop all social interactions including use of public transport, stop commuters entering/leaving town and restricts movement of people.

4. Partial lockdown: Per default is Off. Stops all social activities including public transport, commuters entering and leaving the city are still allowed. No restriction of going out of home.
 
5. Keep social distancing: tell people to keep a minimum set distance in metres from each other. The order is not followed accurately but more randomly and proportionate to the set distance. This variable is only taken into account when complete lock down is set to Off.

6. Prioritise elderly: Per default is Off. When set to On it tells the simulation to prioritise elderly patient (older > 60) access to hospital beds.



## EXTENDING THE MODEL


## NETLOGO FEATURES


## CREDITS AND REFERENCES



## HOW TO CITE


For the model itself:

* Jimenez Romero, C (2020).  NetLogo Covid19 dynamics model.  http://cristianjimenez.org
The Open University, UK

Please cite the NetLogo software as:

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

person lefty
false
0
Circle -7500403 true true 170 5 80
Polygon -7500403 true true 165 90 180 195 150 285 165 300 195 300 210 225 225 300 255 300 270 285 240 195 255 90
Rectangle -7500403 true true 187 79 232 94
Polygon -7500403 true true 255 90 300 150 285 180 225 105
Polygon -7500403 true true 165 90 120 150 135 180 195 105

person righty
false
0
Circle -7500403 true true 50 5 80
Polygon -7500403 true true 45 90 60 195 30 285 45 300 75 300 90 225 105 300 135 300 150 285 120 195 135 90
Rectangle -7500403 true true 67 79 112 94
Polygon -7500403 true true 135 90 180 150 165 180 105 105
Polygon -7500403 true true 45 90 0 150 15 180 75 105

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

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.1.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
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
