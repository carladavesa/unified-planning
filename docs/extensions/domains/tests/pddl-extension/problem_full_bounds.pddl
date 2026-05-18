(define (problem pancake-p1)

  (:domain pancake-int)

  (:init
    ;; initial scrambled stack: [3 1 4 0 2]
    (= (val 0) 1)
    (= (val 1) 8)
    (= (val 2) 9)
    (= (val 3) 6)
    (= (val 4) 7)
    (= (val 5) 5)
    (= (val 6) 3)
    (= (val 7) 0)
    (= (val 8) 2)
    (= (val 9) 4)
  )

  (:goal
    (and
      ;; sorted ascending: [0 1 2 3 4]
      (= (val 0) 0)
      (= (val 1) 1)
      (= (val 2) 2)
      (= (val 3) 3)
      (= (val 4) 4)
      (= (val 5) 5)
      (= (val 6) 6)
      (= (val 7) 7)
      (= (val 8) 8)
      (= (val 9) 9)

    )
  )
)
