import { prisma } from './backend/db';

async function main() {
  console.log('--- Starting Database Tests ---');

  // Test 1: Count intersections
  const intersectionCount = await prisma.intersection.count();
  console.log(`Current intersections in DB: ${intersectionCount}`);

  // Test 2: Create a test intersection
  console.log('Creating a test intersection...');
  const newIntersection = await prisma.intersection.upsert({
    where: { id: 'test_intersection_001' },
    update: {},
    create: {
      id: 'test_intersection_001',
      name: 'Test Intersection',
      google_maps_url: 'https://maps.google.com/?q=test',
    }
  });
  console.log('Intersection created/found:', newIntersection);

  // Test 3: Create a test traffic observation
  console.log('Creating a test traffic observation...');
  const newObservation = await prisma.trafficObservation.create({
    data: {
      intersection_id: newIntersection.id,
      observed_at: new Date(),
      dominant_color: 'green',
      confidence: 0.95,
      extractor_version: 'vtest',
    }
  });
  console.log('Observation created:', newObservation);

  // Test 4: Query back the observation
  const obsFromDb = await prisma.trafficObservation.findUnique({
    where: { id: newObservation.id }
  });
  console.log('Observation queried back:', obsFromDb);

  console.log('--- Database Tests Completed Successfully ---');
}

main()
  .catch((e) => {
    console.error('Test Failed:', e);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
