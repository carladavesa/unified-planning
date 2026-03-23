;; 4,5,3,2,1

(define (problem n5-1)
  (:domain pancake-sorting)
  (:objects
    p1 - position
    p2 - position
    p3 - position
    p4 - position
    p5 - position
    t1 - tile
    t2 - tile
    t3 - tile
    t4 - tile
    t5 - tile
  )
  (:init
    (at t4 p1) 
    (at t5 p2) 
    (at t3 p3) 
    (at t2 p4) 
    (at t1 p5)
    (flipat p2 p1 p2)
    (flipat p2 p2 p1)
    (flipat p3 p1 p3)
    (flipat p3 p2 p2)
    (flipat p3 p3 p1)
    (flipat p4 p1 p4)
    (flipat p4 p2 p3)
    (flipat p4 p3 p2)
    (flipat p4 p4 p1)
    (flipat p5 p1 p5)
    (flipat p5 p2 p4)
    (flipat p5 p3 p3)
    (flipat p5 p4 p2)
    (flipat p5 p5 p1)
    (lte p1 p1)
    (lte p1 p2)
    (lte p1 p3)
    (lte p1 p4)
    (lte p1 p5)
    (lte p2 p2)
    (lte p2 p3)
    (lte p2 p4)
    (lte p2 p5)
    (lte p3 p3)
    (lte p3 p4)
    (lte p3 p5)
    (lte p4 p4)
    (lte p4 p5)
    (lte p5 p5)
    (= (total-cost) 0)
  )
  (:goal 
    (and (at t1 p1) 
    (at t2 p2) 
    (at t3 p3) 
    (at t4 p4) 
    (at t5 p5))
  )
  (:metric minimize (total-cost))
)